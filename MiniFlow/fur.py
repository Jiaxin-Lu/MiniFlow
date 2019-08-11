from MiniFlow.ops import *
import numpy as np

def random_normal(shape, mean=0.0, stddev=1.0, dtype=float32, seed=None, name=None):
    ret = np.random.normal(loc=mean, scale=stddev, size=shape).astype(dtype)
    return ret

class Session(object):
    def __enter__(self):
        return self
    def __exit__(self, e_t, e_v, t_b):
        assert True
    def __call__(self, name="Session"):
        newSession = Session()
        newSession.name = name
        newSession.ex = None
        return newSession
    def run(self, eval_node_list, feed_dict = {}):
        is_list = True
        if not isinstance(eval_node_list, list):
            is_list = False
            eval_node_list = [eval_node_list]
        self.ex = Executor(eval_node_list = eval_node_list)
        if (is_list):
            return self.ex.run(feed_dict = feed_dict)
        else:
            return self.ex.run(feed_dict=feed_dict)[0]

class train(object):
    class Optimizer(object):
        def __init__(self):
            return None
        def get_variables_list(self):
            variables_list = []
            for variable in global_variables:
                variables_list.append(variable)
            return variables_list

    class GradientDescentOptimizer(Optimizer):
        def __init__(self, learning_rate = 0.01, name = "GradientDescentOptimizer"):
            self.learning_rate = learning_rate
            self.name = name
        def minimize(self, eval_node):
            variables_all = self.get_variables_list()
            variables_used = find_topo_sort(node_list=[eval_node])
            variables_list = []
            for v in variables_used:
                if v in variables_all:
                    variables_list.append(v)
            variables_grad = gradients(eval_node, variables_list)
            variables_ans = []
            for i, variable in enumerate(variables_list):
                variables_ans.append(assign(variable, variable - self.learning_rate * variables_grad[i]))
            return list_to_node(variables_ans)

    class AdamOptimizer(Optimizer):
        def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, name = "Adam"):
            self. learning_rate = learning_rate
            self. beta1 = beta1
            self. beta2 = beta2
            self. epsilon = epsilon
            self. name = name
        def minimize(self, eval_node):
            variables_list = self.get_variables_list()
            # variables_used = find_topo_sort(node_list=[eval_node])
            # variables_list = []
            # for v in variables_used:
            #    if v in global_variables:
            #        variables_list.append(v)
            variables_grad = gradients(eval_node, variables_list)
            variables_ans = []

            self.t = constant(0)
            self.t_ = assign(self.t, self.t+1)
            self.lrt = self.learning_rate * sqrt_op(1-pow_op(constant(self.beta2), self.t_)) / (1-pow_op(constant(self.beta1), self.t_))
            self.m = []
            self.m_ = []
            self.v = []
            self.v_ = []
            for variable in variables_list:
                self.m.append(constant(0))
                self.v.append(constant(0))
            
            for i, variable in enumerate(variables_list):
                grad = variables_grad[i]
                new_m = self.m[i]
                m_t = assign(new_m, new_m*self.beta1 + grad*(1-self.beta1))
                new_v = self.v[i]
                v_t = assign(new_v, new_v * self.beta2 + grad * grad * (1-self.beta2))
                newVal = variable - self.lrt*m_t / (sqrt_op(v_t) + constant(self.epsilon))

                variables_ans.append(assign(variable, newVal))
            return list_to_node(variables_ans)

    
class nn(object):
    class SoftmaxOp(Op):
        def __call__(self, node, dim=-1, name = None):
            if name is None:
                name = "Softmax(%s, dim=%s)" % (node.name, dim)
            exp_node = exp(node)
            new_node = exp_node / broadcastto_op(reduce_sum(exp_node, axis = dim), exp_node)
            new_node.name = name
            return new_node
        
    softmax = SoftmaxOp()

    class SoftmaxCrossEntropyWithLogitsOp(Op):
        def __call__(self, logits, labels):
            return (-reduce_sum(labels * log(nn.softmax(logits)), reduction_indices=[1]))
            
    softmax_cross_entropy_with_logits = SoftmaxCrossEntropyWithLogitsOp()
    def relu(node):
        return relu_op(node, node)

    class DropoutOp(Op):
        def __call__(self, node_A, node_B, name = None):
            new_node = mul_op(node_A, shapeofprob_op(node_A, node_B)) / node_B
            if name is None:
                name = "Dropout(%s, prob = %s)" % (node_A.name, node_B.name)
            new_node.name = name
            return new_node
    dropout = DropoutOp()
    max_pool = max_pool_op
    conv2d = conv2d_op
