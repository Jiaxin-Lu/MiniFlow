#define DLLEXPORT extern "C"

//#include "framework.h"
#include <cstdio>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cblas.h>
#include <cstring>

typedef unsigned int uint;
using std::max;
using std::min;

const float aa = 1.0;
const float bb = 0.0;
DLLEXPORT
void matmul(float *a, float *b, float *c, int na, int ma, int nb, int mb, bool transA, bool transB)
{
	if (transA)
	{
		if (transB)
			cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, ma, nb, mb, aa, a, ma, b, mb, bb, c, nb);
		else
			cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ma, mb, na, aa, a, ma, b, mb, bb, c, mb);
	}
	else
	{
		if (transB)
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, na, nb, mb, aa, a, ma, b, mb, bb, c, nb);
		else
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, na, mb, ma, aa, a, ma, b, mb, bb, c, mb);
	}
}

inline void mm(const float *A, const float *B, float *C,
			   int n, int k, int m, bool transA, bool transB)
{
	const CBLAS_TRANSPOSE TA = transA ? CblasTrans : CblasNoTrans;
	const CBLAS_TRANSPOSE TB = transB ? CblasTrans : CblasNoTrans;
	// const int lda = transA ? n : k;
	// const int ldb = transB ? k : m;
	// const int ldc = m;
	cblas_sgemm(CblasRowMajor, TA, TB, n, m, k, aa, A, transA ? n : k, B, transB ? k : m, bb, C, m);
}
float *image = nullptr;
int memory_size = 0;
DLLEXPORT
void conv2d(const float *init, const float *filter, float *result,
			int bch, int ih, int iw, int fn, int fm, int fc, int rc, int oh, int ow)
{
	register int size = oh * ow * fn * fm * fc, rows = iw - fn + 1, fsize = fn * fm * fc;
	register int fmfc = fc*fm, ohow = oh*ow;

	//float *image = (float*)malloc(size*sizeof(float));
	if (size > memory_size)
	{
		free(image);
		image = (float *)malloc(size * sizeof(float));
		memory_size = size;
	}
	for (int b = 0; b < bch; ++b)
	{
		memset(image, 0, size * sizeof(float));
		const float *img_now = init + b * ih * iw * fc;
		for (int i = 0; i <= ih - fn; ++i)
		{
			for (int j = 0; j <= iw - fm; ++j)
			{
				float *tmp_now = image + (i * rows + j) * fsize;
				const float *img_now_ = img_now+j*fc;
				int ad = 0;
				for (int x = i; x < i + fn; ++x)
				{
					memcpy(tmp_now + ad, img_now_ + x*iw*fc, fmfc*sizeof(float));
					// memcpy(tmp_now + ad, img_now + (x * iw + j) * fc, fcfm * sizeof(float));
					ad += fmfc;
				}
			}
		}
		mm(image, filter, result + b * ohow * rc, ohow, fsize, rc, false, false);
	}
	//free(image);
}

DLLEXPORT
void conv2d_grad2(const float *init, const float *filter, float *result,
				  int bch, int ih, int iw, int fn, int fm, int fc, int rc, int oh, int ow)
{
	register int size = oh * ow * fn * fm * fc, rows = iw - fn + 1, fsize = fn * fm * fc * rc;
	register int fmfc = fm*fc, ohow = oh*ow;
	memset(result, 0, fsize * sizeof(float));
	//for (int i=0;i<fn*fm*fc*rc;++i) result[i] = 0;
	//float *image = (float*)malloc(size * sizeof(float));
	if (size > memory_size)
	{
		free(image);
		image = (float *)malloc(size * sizeof(float));
		memory_size = size;
	}

	float *tmp = (float *)malloc(fsize * sizeof(float));
	for (int b = 0; b < bch; ++b)
	{
		memset(image, 0, sizeof(float) * size);
		const float *img_now = init + b * ih * iw * fc;
		for (int i = 0; i <= ih - fn; ++i)
		{
			for (int j = 0; j <= iw - fm; ++j)
			{
				float *tmp_img = image + (i * rows + j) * fn * fm * fc;
				const float *img_now_ = img_now + j*fc;
				int ad = 0;
				for (int x = i; x < i + fn; ++x)
				{
					//for (int y=j;y<j+fm;++y,ad+=fc)
					//memcpy(tmp_img+ad, img_now+(x*iw+y)*fc,fc*sizeof(float));
					memcpy(tmp_img + ad, img_now_ + x*iw*fc, fmfc*sizeof(float));
					// memcpy(tmp_img + ad, img_now + (x * iw + j) * fc, fm * fc * sizeof(float));
					ad += fmfc;
				}
			}
		}
		memset(tmp, 0, fsize * sizeof(float));
		mm(image, filter + b * ohow * rc, tmp, fn * fmfc, ohow, rc, true, false);
		for (int i = 0; i < fsize; ++i)
			*(result + i) += *(tmp + i);
	}
	free(tmp);
	//free(image);
}

DLLEXPORT
int max_pool_gradient(float *g, int bch, int gh, int gw, int ic, float *output, int hs, int ws, float *z, int zh, int zw)
{
	int gbs = gh * gw * ic;
	int gghs = gw * ic;
	int zbs = zh * zw * ic;
	int zghs = hs * zw * ic;
	int zgws = ws * ic;
	int zhs = zw * ic;
	for (int b = 0; b < bch; ++b)
	{
		for (int i = 0; i < gh; ++i)
		{
			for (int j = 0; j < gw; ++j)
			{
				float *g_gw = g + b * gbs + i * gghs + j * ic;
				for (int c = 0; c < ic; ++c)
				{
					float *z_ic = z + b * zbs + i * zghs + j * zgws + c;
					float *max_l = z_ic;
					for (int di = 0; di < hs; ++di)
					{
						for (int dj = 0; dj < ws; ++dj)
						{
							register float *mat_a = z_ic + di * zhs + dj * ic;
							if ((*mat_a) > (*max_l))
								max_l = mat_a;
						}
					}
					output[max_l - z] += g_gw[c];
				}
			}
		}
	}
	return 0;
}