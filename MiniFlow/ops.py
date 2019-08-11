import numpy as np
# import MiniFlow.c_ops as c_ops
# from MiniFlow.topo import *

global_variables = {}

float32 = np.float32
float64 = np.float64
float16 = np.float16
float_ = np.float_
int64 = np.int64
int32 = np.int32
int16 = np.int16
int8 = np.int8
int_ = np.int_
bool_ = np.bool_
zeros = np.zeros
"""
C PARTS
"""
import ctypes
import os
import sys
import platform

ndpointer = np.ctypeslib.ndpointer
c_float32 = ctypes.c_float
c_int32 = ctypes.c_int32
c_bool = ctypes.c_bool
if platform.system() == 'Linux':
    cur_path = sys.path[0]
    dll_path = os.path.join(cur_path, "MiniFlow", "kernel.so")
    c_kernel = ctypes.CDLL(dll_path)
else:
    cur_path = os.path.dirname(__file__)
    dll_path = os.path.join(cur_path, "kernel", "x64", "Release", "kernel.dll")
    c_kernel = ctypes.CDLL(dll_path)

def get_pointer(input):
    return input.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
###

class Node(object):
    """Node in a computation graph."""
    def __init__(self):
        """Constructor, new node is indirectly created by Op object __call__ method.
            
            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.op: the associated op object, 
                e.g. add_op object if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant,
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging purposes.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""

    def __add__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            # Add by a constant stores the constant in the new node's const_attr field.
            # 'other' argument is a constant
            new_node = add_op(self, constant(other))
        return new_node

    def __mul__(self, other):
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            new_node = mul_op(self, constant(other))
        return new_node

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return neg_op(self)
    
    def __sub__(self, other):
        if isinstance(other, Node):
            new_node = sub_op(self, other)
        else:
            new_node = sub_op(self, constant(other))
        return new_node
    
    def __rsub__(self, other):
        if isinstance(other, Node):
            new_node = sub_op(other, self)
        else:
            new_node = sub_op(constant(other), self)
        return new_node

    def __truediv__(self, other):
        if isinstance(other, Node):
            new_node = div_op(self, other)
        else:
            new_node = div_op(self, constant(other))
        return new_node
    
    def __rtruediv__(self, other):
        if isinstance(other, Node):
            new_node = div_op(other, self)
        else:
            new_node = div_op(constant(other), self)
        return new_node

    __floordiv__ = __truediv__
    __rfloordiv__ = __rtruediv__
    __div__ = __truediv__
    __rdiv__= __rtruediv__

    def __str__(self):
        """Allow print to display node name.""" 
        return self.name

    __repr__ = __str__
    
    def eval(self, feed_dict={}):
        """Calculate the value of this node"""
        ex = Executor(eval_node_list=[self])
        return ex.run(feed_dict=feed_dict)[0]

    run = eval

class Op(object):
    """Op represents operations performed on nodes."""
    def __call__(self):
        """Create a new node and associate the op object with the node.
        
        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.

        Returns
        -------
        An output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node, output_grad):
        """Given value of output gradient, compute gradient contributions to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: value of output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError
#DONE!
class VariablesInitOp(Op):
    def __call__(self):
        """Feed the global 'variables' into the exact variables."""
        new_node = Op.__call__(self)
        new_node.inputs = []
        new_node.name = "Global_Variables_Initializer"
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 0
        for key, value in global_variables.items():
            if isinstance(value, Node):
                key.const_attr = value.const_attr
            else:
                key.const_attr = value
        return 0 #success

    def gradient(self, node, output_grad):
        raise NotImplementedError
#DONE!
class VariableOp(Op):
    def __call__(self, initial_value, dtype = None, shape = None, name = "Variable"):
        """Create a variable node"""
        new_node = Op.__call__(self)
        if shape is not None:
            assert shape == initial_value.shape
        if dtype is not None:
            if isinstance(initial_value, np.ndarray):
                global_variables[new_node] = initial_value.astype(dtype)
            else:
                global_variables[new_node] = np.array(initial_value).astype(dtype)
        else:
            global_variables[new_node] = initial_value
        new_node.name = name
        return new_node

    def compute(self, node, input_vals):
        if node.const_attr is None:
            raise UnboundLocalError
        return node.const_attr

    def gradient(self, node, output_grad):
        return None

#DONE!
class NegOp(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "(-%s)" % (node.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return -input_vals[0]
    
    def gradient(self, node, output_grad):
        return [adapt(constant(0.)-output_grad, node.inputs[0])]
#DONE!
class AddOp(Op):
    """Op to element-wise add two nodes."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise addition."""
        assert len(input_vals) == 2
        return input_vals[0] + input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contributions to each input."""
        return [adapt(output_grad, node.inputs[0]), adapt(output_grad, node.inputs[1])]
#DONE!
class SubOp(Op): 
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s-%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] - input_vals[1]

    def gradient(self, node, output_grad):
        return [adapt(output_grad, node.inputs[0]), adapt(constant(0.)-output_grad, node.inputs[1])]
#DONE!
class MulOp(Op):
    """Op to element-wise multiply two nodes."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise multiplication."""

        assert len(input_vals) == 2
        return input_vals[0] * input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input."""
        return [adapt(output_grad*node.inputs[1], node.inputs[0]), adapt(output_grad*node.inputs[0], node.inputs[1])]
#DONE!
class DivOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s/%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] / input_vals[1]

    def gradient(self, node, output_grad):
        return [adapt(output_grad / node.inputs[1], node.inputs[0]), 
            adapt( constant(-1) * output_grad * node.inputs[0] / (node.inputs[1] * node.inputs[1]), node.inputs[1])]

#DONE?
class MatMulOp(Op):
    """Op to matrix multiply two nodes."""
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        """Create a new node that is the result a matrix multiple of two input nodes.

        Parameters
        ----------
        node_A: lhs of matrix multiply
        node_B: rhs of matrix multiply
        trans_A: whether to transpose node_A
        trans_B: whether to transpose node_B

        Returns
        -------
        Returns a node that is the result a matrix multiple of two input nodes.
        """
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (node_A.name, node_B.name, str(trans_A), str(trans_B))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, return result of matrix multiplication."""
        assert len(input_vals) == 2
        mat_A = input_vals[0].astype(float32)
        mat_B = input_vals[1].astype(float32)
        # if node.matmul_attr_trans_A == True:
        #     mat_A = mat_A.T
        # if node.matmul_attr_trans_B == True:
        #     mat_B = mat_B.T

        na, ma = mat_A.shape
        nb, mb = mat_B.shape
        if node.matmul_attr_trans_A:
            if node.matmul_attr_trans_B:
                result = np.ndarray(shape = (ma, nb), dtype = float32)
            else:
                result = np.ndarray(shape = (ma, mb), dtype = float32)
        else:
            if node.matmul_attr_trans_B:
                result = np.ndarray(shape = (na, nb), dtype = float32)
            else:
                result = np.ndarray(shape = (na, mb), dtype = float32)
        # result = np.zeros(shape=(na, mb), dtype=float32)

        #print(node.matmul_attr_trans_A, node.matmul_attr_trans_B)
        #print("in matmul")
        #print(na, ma, nb, mb)
        #print("mata")
        #print(mat_A)
        #print("matb")
        #print(mat_B)
        matmul_c(get_pointer(mat_A),get_pointer(mat_B),get_pointer(result),na,ma,nb,mb,node.matmul_attr_trans_A,node.matmul_attr_trans_B)
        #print("result")
        #print(result)
        #print(result)
        #print("out matmul")
        # result = np.dot(mat_A,mat_B)
        #np.testing.assert_allclose(result, result1, atol=1e-1)
        return result

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input.
            
        Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
        """
        return [matmul_op(output_grad, node.inputs[1], False, True),
                matmul_op(node.inputs[0], output_grad, True, False)]

matmul_c = c_kernel.matmul
#DONE!
class LogOp(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "log(%s)" % node.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.log(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad / node.inputs[0]]

#DONE!
class ExpOp(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "exp(%s)" % node.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.exp(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * exp_op(node.inputs[0])]

#DONE!
class SqrtOp(Op):
    def __call__(self, node):
        """Creates a node that represents np.sqrt(node_A)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "Sqrt(%s)" % (node.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        output_val = np.sqrt(input_vals[0])
        return output_val

    def gradient(self, node, output_grad):
        raise NotImplementedError

#DONE!
class PowOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "Pow(%s, %s)" % (node_A.name, node_B.name)
        return new_node
    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return np.power(input_vals[0], input_vals[1])
    def gradient(self, node, output_grad):
        raise NotImplementedError
#DONE!
class ReshapeOp(Op):
    def __call__(self, node, shape):
        """reshape node to shape"""
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "reshape(%s, shape = %s)" % (node.name, shape)
        new_node.const_attr = shape
        return new_node
    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.reshape(input_vals[0], tuple(node.const_attr))
    def gradient(self, node, output_grad):
        return [reshape_node(output_grad, node.inputs[0])]

#DONE!
class ReshapeNodeOp(Op):
    def __call__(self, node_A, node_B):
        """reshape node_A to the shape of node_B"""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "reshape(%s, shape = %s.shape)" % (node_A.name, node_B.name)
        return new_node
    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return np.reshape(input_vals[0], input_vals[1].shape)
    def gradient(self, node, output_grad):
        return [reshape_node(output_grad,node.inputs[0]), zeroslike_op(node.inputs[1])]

#DONE!
class BroadcastToOp(Op):
    def __call__(self, node_A, node_B):
        """np.broadcast_to(node_A, node_B.shape)"""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A,node_B]
        new_node.name = "broadcast_to(%s,%s.shape)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        output_val = input_vals[0]
        if (len(output_val.shape) < len(input_vals[1].shape)):
            front_align = True
            for dim, in_size in enumerate(output_val.shape):
                if input_vals[1].shape[dim] != in_size:
                    front_align = False
                    break
            new_shape = output_val.shape
            if front_align:
                while len(new_shape) < len(input_vals[1].shape):
                    new_shape = new_shape+(1,)
            output_val.resize(new_shape)
        output_val = np.broadcast_to(output_val,input_vals[1].shape)
        return output_val

    def gradient(self, node, output_grad):
        grad_A = reducesum_shape_op(output_grad, node.inputs[0])
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

#DONE!
class ReducesumShapeOp(Op):
    """adapt sum(node_A) to the shape of node_B"""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A,node_B]
        new_node.name = "reducesum_shape(%s,%s)" % (node_A.name,node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        shape_A = input_vals[0]
        shape_B = input_vals[1]
        while (len(shape_A.shape) > len(shape_B.shape)):
            shape_A = np.sum(shape_A, axis=0)
        for dim in range(len(shape_A.shape)):
            if (shape_A.shape[dim] > shape_B.shape[dim]):
                assert shape_B.shape[dim] == 1
                shape_A = np.sum(shape_A, axis = dim, keepdims=True)
        return shape_A

    def gradient(self, node, output_grad):
        return [broadcastto_op(output_grad, node.inputs[0]), zeroslike_op(node.inputs[1])]

#DONE!
class ReducemeanShapeOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A,node_B]
        new_node.name = "reducemean_shape(%s,%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        shape_A = input_vals[0]
        shape_B = input_vals[1]
        while (len(shape_A.shape) > len(shape_B.shape)):
            shape_A = np.mean(shape_A, axis = 0)
        for dim in range(len(shape_A.shape)):
            if (shape_A.shape[dim] > shape_B.shape[dim]):
                assert shape_B.shape[dim] == 1
                shape_A = np.mean(shape_A, axis=dim, keepdims=True)
        return shape_A

    def gradient(self, node, output_grad):
        raise NotImplementedError

#DONE!
class ReduceSumOp(Op):
    def __call__(self, node, axis=None, keep_dims=False, reduction_indices=None):
        new_node = Op.__call__(self)
        if axis is None and reduction_indices is not None:
            axis = tuple(reduction_indices)
        new_node.inputs = [node]
        new_node.name = "reduce_sum(%s, axis=%s, keep_dims=%s)" % (node.name, axis, keep_dims)
        new_node.const_attr = (axis, keep_dims)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.sum(input_vals[0],axis=node.const_attr[0],keepdims=node.const_attr[1])

    def gradient(self, node, output_grad):
        return [broadcastto_op(output_grad, node.inputs[0])]

#DONE!
class ReduceMeanOp(Op):
    def __call__(self, node, axis = None, keep_dims = False, reduction_indices = None):
        new_node = Op.__call__(self)
        if axis is None and reduction_indices is not None:
            axis = tuple(reduction_indices)
        new_node.inputs = [node]
        new_node.name = "reduce_mean(%s, axis=%s, keep_dims=%s)" % (node.name, axis, keep_dims)
        new_node.const_attr = (axis, keep_dims)
        return new_node
        
    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.mean(input_vals[0], axis=node.const_attr[0], keepdims=node.const_attr[1])
    
    def gradient(self, node, output_grad):
        return [adapt(broadcastto_op(output_grad, node.inputs[0]) / \
                reduce_sum(oneslike_op(node.inputs[0]),axis=node.const_attr[0],keep_dims=node.const_attr[1]),node.inputs[0])]

#DONE!
class CastOp(Op):
    def __call__(self, node, dtype = float64, name=None):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.const_attr = dtype
        if name is None:
            new_node.name = "cast(%s, dtype = %s)" % (node.name, str(dtype))
        else:
            new_node.name = name
        return new_node
    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return input_vals[0].astype(node.const_attr)
    def gradient(self, node, output_grad):
        return [output_grad]

#DONE!
class EqualOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A,node_B]
        new_node.name = "[%s == %s]" % (node_A.name,node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return np.equal(input_vals[0], input_vals[1])

    def gradient(self, node, output_grad):
        raise NotImplementedError    

#DONE!
class ArgmaxOp(Op):
    def __call__(self, node, axis = None, name = None):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.const_attr = axis
        if name is None:
            new_node.name = "argmax(%s, axis = %s)" % (node.name, axis)
        else:
            new_node.name = name
        return new_node
    
    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.argmax(input_vals[0], axis = node.const_attr)
    
    def gradient(self, node, output_grad):
        raise NotImplementedError
    
#DONE!
class AssignOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        if not isinstance(node_B, Node):
            node_B = constant(node_B)
        new_node.inputs = [node_B]
        new_node.const_attr = node_A
        new_node.name = "(%s = %s)" % (node_A.name, node_B.name)
        return new_node
    
    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        assert isinstance(node.const_attr.op, VariableOp) or isinstance(node.const_attr.op, ConstantOp)
        node.const_attr.const_attr = input_vals[0]
        return input_vals[0]

    def gradient(self, node, output_grad):
        raise NotImplementedError

#DONE!
class ConstantOp(Op):
    def __call__(self, initial_value, dtype = None, shape = None, name = "Const"):
        new_node = Op.__call__(self)
        if not isinstance(initial_value, np.ndarray) and (shape is not None):
            initial_value = np.ones(shape=shape) * initial_value
        new_node.const_attr = np.array(initial_value).reshape(shape).astype(dtype)
        new_node.name = name
        return new_node

    def compute(self, node, input_vals):
        return node.const_attr

    def gradient(self, node, output_grad):
        return None

#DONE!
class PlaceholderOp(Op):
    """Op to feed value to a nodes."""
    def __call__(self, dtype, shape = None, name = "Placeholder"):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        new_node.const_attr = (shape, dtype)
        new_node.name = name
        return new_node

    def compute(self, node, input_vals):
        """No compute function since node value is fed directly in Executor."""
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        """No gradient function since node has no inputs."""
        return None

#DONE!
class ZerosLikeOp(Op):
    """Op that represents a constant np.zeros_like."""
    def __call__(self, node_A):
        """Creates a node that represents a np.zeros array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns zeros_like of the same shape as input."""
        #assert(isinstance(input_vals[0], np.ndarray))
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

#DONE!
class OnesLikeOp(Op):
    """Op that represents a constant np.ones_like."""
    def __call__(self, node_A):
        """Creates a node that represents a np.ones array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns ones_like of the same shape as input."""
        #assert(isinstance(input_vals[0], np.ndarray))
        input_vals[0] = np.array(input_vals[0])
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

#DONE! 
class ListToNodeOp(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = node
        new_node.name = "list_to_node(%s)" % (str([n.name for n in node]))
        return new_node

    def compute(self, node, input_vals):
        return None

    def gradient(self, node, output_grad):
        raise NotImplementedError

#DONE!
def softmax_func(y):
    expy = np.exp(y-np.max(y,axis=-1,keepdims=True))
    softmax = expy / np.sum(expy,axis=-1,keepdims=True)
    return softmax
  
#DONE!
class SoftmaxOp(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "Softmax(%s)" % (node.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return softmax_func(input_vals[0])

    def gradient(self, node, output_grad):
        raise NotImplementedError

#DONE!
class SoftmaxCrossEntropyOp(Op):
    def __call__(self,node_A,node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A,node_B]
        new_node.name = "SoftmaxCrossEntropy(%s, %s)" % (node_A.name, node_B.name)
        return new_node
    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        y = input_vals[0]
        y_ = input_vals[1]
        softmax = softmax_func(y)
        return -np.sum(y_*np.log(softmax), axis = -1, keepdims=True)
    def gradient(self, node, output_grad):
        grad_A = (softmax_op(node.inputs[0]) - node.inputs[1]) * output_grad
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

#DONE!
class ReluOp(Op):
    def __call__(self,node_A,node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A,node_B]
        new_node.name = "Relu(%s, %s)" % (node_A.name, node_B.name)
        return new_node
    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return (np.greater_equal(input_vals[1],0)).astype(np.int32) * input_vals[0]
    def gradient(self, node, output_grad):
        return [relu_op(output_grad, node.inputs[1]), zeroslike_op(node.inputs[1])]

#DONE!
class ShapeOfProbOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A,node_B]
        new_node.name = "Shape_of_prob(shape = %s, prob = %s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return (np.random.uniform(size=input_vals[0].shape) < input_vals[1])
        
    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0]), zeroslike_op(node.inputs[1])]


def Conv2d_Cal(_input, _filter):
    input = _input
    filter = _filter
    bch, n, m, fc = input.shape
    fn, fm, fc, rc = filter.shape
    u = (fn-1)//2
    v = (fm-1)//2
    input_pad = np.pad(input, ((0,0),(u,u),(v,v),(0,0)),"constant")
    bch, ih, iw, fc = input_pad.shape
    input_pad = input_pad.astype(float32)
    col_filter = filter.reshape(-1, rc).astype(np.float32)
    result = np.zeros(shape=(bch,n,m,rc),dtype=np.float32)
    c_kernel.conv2d(get_pointer(input_pad), get_pointer(col_filter), get_pointer(result),
        bch, ih, iw, fn, fm, fc, rc, n, m)
    return result

class Conv2dOp(Op):
    def __call__(self, input, filter, strides=[1, 1, 1, 1], padding='SAME', name=None):
        assert padding == 'SAME'  # 'VALID' not supported
        new_node = Op.__call__(self)
        new_node.inputs = [input, filter]
        new_node.const_attr = (strides, padding)
        if name is None:
            new_node.name = "Conv2D(%s,filter=%s)" % (input.name, filter.name)
        else:
            new_node.name = name
        return new_node

    # profile
    def compute(self, node, input_vals):
        input=input_vals[0]
        filter=input_vals[1]
        return Conv2d_Cal(input, filter)

    def gradient(self, node, output_grad):
        return [conv2d_grad_op1(node.inputs[0], node.inputs[1], output_grad, node.const_attr),
                conv2d_grad_op2(node.inputs[0], node.inputs[1], output_grad, node.const_attr)]

def fz(a):
    return a[::-1]
def FZ(mat):
    return np.array(fz(list(map(fz,mat))))
class Conv2DGradientOp1(Op):
    def __call__(self,  input, filter, node_grad, stridesAndPadding):
        new_node = Op.__call__(self)
        new_node.inputs = [input, filter, node_grad]
        new_node.const_attr = stridesAndPadding
        return new_node

    # profile
    def compute(self, node, input_vals):
        input = input_vals[0]
        filter = input_vals[1]
        grad = input_vals[2]
        filter = FZ(filter[:][:][:][:])
        filter = np.swapaxes(filter, 2, 3)
        return Conv2d_Cal(grad, filter)

    def gradient(self, node, output_grad):
        raise NotImplementedError

class Conv2DGradientOp2(Op):
    def __call__(self, input, filter, node_grad, stridesAndPadding):
        new_node = Op.__call__(self)
        new_node.inputs = [input, filter, node_grad]
        new_node.const_attr = stridesAndPadding
        return new_node

    # profile
    def compute(self, node, input_vals):
        input=input_vals[0]
        grad=input_vals[2]
        filter=input_vals[1]

        bch, n, m, ic = input.shape
        fn, fm, fc, rc = filter.shape
        u = (fn-1)//2
        v = (fm-1)//2
        input_pad = np.pad(input, ((0,0),(u,u),(v,v),(0,0)),"constant")
        input_pad = input_pad.astype(float32)
        bch, ih, iw, fc = input_pad.shape
        result = np.zeros(shape=(fn*fm*fc,rc), dtype=np.float32)
        col_grad = np.reshape(grad,[bch,-1,rc]).astype(np.float32)
        #print("input_pad")
        #print(input_pad)
        #print("col_grad")
        #print(col_grad)
        c_kernel.conv2d_grad2(get_pointer(input_pad), get_pointer(col_grad), get_pointer(result),
            bch, ih, iw, fn, fm, fc, rc, n, m)
        #print("result")
        #print(result)
        return result.reshape(filter.shape)

    def gradient(self, node, output_grad):
        raise NotImplementedError

conv2d_grad_op1 = Conv2DGradientOp1()
conv2d_grad_op2 = Conv2DGradientOp2()
conv2d_op = Conv2dOp()
def get_patch(ori, i, j, f_h, f_w, strides, i_c=None):
    if i_c is None:
        return ori[:, i * strides[1]:i * strides[1] + f_h, j * strides[2]:j * strides[2] + f_w, :]
    else:
        return ori[:, i * strides[1]:i * strides[1] + f_h, j * strides[2]:j * strides[2] + f_w, i_c]
def zero_padding_func(ori, up, down, left, right):
    ret = np.zeros([ori.shape[0], ori.shape[1] + up + down, ori.shape[2] + left + right, ori.shape[3]])
    ret[:, up:up + ori.shape[1], left:left + ori.shape[2], :] = ori[:, :, :, :]
    return ret
class MaxPoolOp(Op):
    def __call__(self, value, ksize, strides, padding = "SAME" , dtype = float32):
        new_node = Op.__call__(self)
        new_node.const_attr = (ksize, strides)
        new_node.dtype = dtype
        new_node.inputs = [value]
        new_node.name = "max_pool(%s)" % (value.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise addition."""
        strides = node.const_attr[1]
        ksize = node.const_attr[0]
        input = input_vals[0]
        ishape = list(input_vals[0].shape)
        zh = ((ishape[1]-1) // strides[1])*strides[1] + ksize[1]
        zw = ((ishape[2]-1) // strides[2])*strides[2] + ksize[2]
        z = zero_padding_func(ori = input, up=(zh-ishape[1])//2, down=(zh-ishape[1]+1)//2, left=(zw-ishape[2])//2, right=(zw-ishape[2]+1)//2)

        output = np.zeros([ishape[0],(ishape[1]-ksize[1])//strides[1]+1,(ishape[2]-ksize[2])//strides[2]+1,ishape[3]])
        oshape = output.shape
        for i in range(oshape[1]):
            for j in range(oshape[2]):
                output[:,i,j,:] = np.max(
                    get_patch(z,i,j,ksize[1],ksize[2],strides), axis=(1,2)
                )
                #np.amax(input[:,i*strides[1]:(i+1)*strides[1],j*strides[1]:(j+1)*strides[1],:],axis=(1,2))
        return output
        
    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contribution to input."""
        return [max_pool_grad_op(node.inputs[0], output_grad, node.const_attr[0], node.const_attr[1])]



class MaxPoolGradOp(Op):
    def __call__(self, value, output_grad, ksize, strides, padding = "SAME" , dtype = float32):
        new_node = Op.__call__(self)
        new_node.const_attr = (ksize, strides)
        new_node.dtype = dtype
        new_node.inputs = [value, output_grad]
        new_node.name = "max_pool_grad(%s)" % (value.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise addition."""
        
        strides = node.const_attr[1]
        ksize = node.const_attr[0]
        ishape = list(input_vals[0].shape)
        input = input_vals[0]
        z_h = ((ishape[1] - 1) // strides[1]) * strides[1] + ksize[1]
        z_w = ((ishape[2] - 1) // strides[2]) * strides[2] + ksize[2]
        z = zero_padding_func(ori = input, up = (z_h - ishape[1])//2, down = (z_h-ishape[1]+1)//2, left=(z_w-ishape[2])//2, right=(z_w-ishape[2]+1)//2)
        output_val = np.zeros((ishape[0], z_h, z_w, ishape[3]), dtype=np.float32)
        output_grad = input_vals[1]

        g = output_grad.astype(np.float32)
        input32 = z.astype(np.float32)
        gshape = list(output_grad.shape)
        assert c_kernel.max_pool_gradient(get_pointer(g), gshape[0], gshape[1], gshape[2], gshape[3],
                                    get_pointer(output_val), ksize[1], ksize[2], get_pointer(input32), z.shape[1], z.shape[2]) == 0
        # c_ops.max_pool_gradient(gradient = output_grad, input = z, output = output_val, ksize=ksize, strides=strides)
        up = (z_h-ishape[1])//2
        left = (z_w-ishape[2])//2
        output_val = output_val[:,up:up+ishape[1],left:left+ishape[2],:]
        return output_val

    def gradient(self, node, output_grad):
        raise NotImplementedError

max_pool_grad_op = MaxPoolGradOp()
max_pool_op = MaxPoolOp()

# Create global singletons of operators.
add_op = AddOp()
mul_op = MulOp()
sub_op = SubOp()
neg_op = NegOp()
div_op = DivOp()
matmul_op = MatMulOp()
log_op = LogOp()
exp_op = ExpOp()
sqrt_op = SqrtOp()
pow_op = PowOp()
placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()
softmax_op = SoftmaxOp()
softmax_cross_entropy_op = SoftmaxCrossEntropyOp()
relu_op = ReluOp()
broadcastto_op = BroadcastToOp()
#adapt = AdaptToOp()
adapt = ReducesumShapeOp()
reducesum_shape_op = ReducesumShapeOp()
reducemean_shape_op = ReducemeanShapeOp()
reshape_node = ReshapeNodeOp()
list_to_node = ListToNodeOp()
shapeofprob_op = ShapeOfProbOp()

assign = AssignOp()
reshape = ReshapeOp()
reduce_sum = ReduceSumOp()
reduce_mean = ReduceMeanOp()
assign = AssignOp()
matmul = MatMulOp()
argmax = ArgmaxOp()
cast = CastOp()
equal = EqualOp()
constant = ConstantOp()
global_variables_initializer = VariablesInitOp()
placeholder = PlaceholderOp()
Variable = VariableOp()

#DONE!
def exp(val):
    if isinstance(val, Node):
        return exp_op(val)
    return np.exp(val)

#DONE!
def log(val):
    if isinstance(val, Node):
        return log_op(val)
    return np.log(val)

#DONE!
class Executor:
    """Executor computes values for a given subset of nodes in a computation graph.""" 
    def __init__(self, eval_node_list):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        """
        self.eval_node_list = eval_node_list
        self.topo_order = find_topo_sort(self.eval_node_list)

    def run(self, feed_dict):
        """Computes values of nodes in eval_node_list given computation graph.
        Parameters
        ----------
        feed_dict: list of variable nodes whose values are supplied by user.

        Returns
        -------
        A list of values for nodes in eval_node_list. 
        """
        node_to_val_map = {}
        for node, value in feed_dict.items():
            node_to_val_map[node] = np.array(value)

        for node in self.topo_order:
            if node in node_to_val_map:
                continue
            vals = [node_to_val_map[i] for i in node.inputs]
            
            compute_val = node.op.compute(node, vals)
            """if (isinstance(compute_val, np.ndarray)):
                node_to_val_map[node] = compute_val
            else:
               node_to_val_map[node] = np.array(compute_val)"""
            node_to_val_map[node] = compute_val

        # Collect node values.
        node_val_results = [node_to_val_map[node] for node in self.eval_node_list]
        return node_val_results

#DONE!
def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """

    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list = {}
    # Special note on initializing gradient of output_node as oneslike_op(output_node):
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    # a map from node to the gradient of that node
    node_to_output_grad = {}
    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = reversed(find_topo_sort([output_node]))


    for node in reverse_topo_order:
        grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = grad

        input_grads = node.op.gradient(node, grad)

        # add input_grads to node_to_grad
        for i in range(len(node.inputs)):
            input_node = node.inputs[i]
            this_node_to_grad = node_to_output_grads_list.get(input_node, [])
            this_node_to_grad.append(input_grads[i])
            node_to_output_grads_list[input_node] = this_node_to_grad

    # Collect results for gradients requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list

def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)


#############
# TOPO SORT #
#############

def find_topo_sort(node_list):
    """Given a list of nodes, return a topological sort list of nodes ending in them.
    
    A simple algorithm is to do a post-order DFS traversal on the given nodes, 
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)