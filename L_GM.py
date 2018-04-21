import warnings
warnings.filterwarnings('ignore')
import numpy as np
import mxnet.ndarray as nd
import mxnet as mx
from mxnet import gluon, autograd


class L_GM_Loss(gluon.nn.HybridBlock):
    def __init__(self, num_class, feature_dim, alpha, lamda, **kwargs):
        super(L_GM_Loss, self).__init__(**kwargs)
        self._num_class = num_class
        self._feature_dim = feature_dim
        self._alpha = alpha
        self._lamda = lamda
        self.mean = self.params.get('mean', shape=(num_class, feature_dim), init=mx.init.Xavier())
        self.var = self.params.get('var', shape=(num_class, feature_dim), init=mx.init.Constant(1))   
    def _classification_probability(self, F, x, y, mean, var):
        batch_size = x.shape[0]
        reshape_var = F.reshape(var, (-1, 1, self._feature_dim))
        reshape_mean = F.reshape(mean, (-1, 1, self._feature_dim))
        expand_data = F.expand_dims(x, 0)
        data_mins_mean = expand_data - reshape_mean
        pair_m_distance = F.batch_dot(data_mins_mean/(reshape_var+1e-8), data_mins_mean, transpose_b=True)/2
        index = F.array([i for i in range(batch_size)])
        m_distance = pair_m_distance[:,index,index].T
        det = F.prod(var, 1)
        label_onehot = F.one_hot(y, self._num_class)
        adjust_m_distance = m_distance + label_onehot*self._alpha*m_distance
        probability = F.exp(-adjust_m_distance)/(F.sqrt(det) + 1e-8)
        return probability, m_distance
    def _regularization_term(self, F, m_distance, y, var):
        batch_var = F.take(var, y)
        batch_det = F.prod(batch_var, 1)
        label_onehot = F.one_hot(y, self._num_class)
        class_distance = F.sum(label_onehot*m_distance, 1)
        # hard to optimizate 
        # likelihood_loss = self._lamda*(class_distance + F.log(batch_det+1e-8)/2)
        likelihood_loss = self._lamda*class_distance
        return likelihood_loss
    def _classification_loss(self, F, probability, y):
        label_onehot = F.one_hot(y, self._num_class)
        class_probability = F.sum(label_onehot*probability, 1)
        classification_loss = -F.log(class_probability/(F.sum(probability, 1)+1e-8)+1e-8)
        return classification_loss
    def hybrid_forward(self,F, x, y, mean, var):
        probability, m_distance = self._classification_probability(F, x, y, mean, var)
        classification_loss = self._classification_loss(F, probability, y)
        likelihood_loss = self._regularization_term(F, m_distance, y, var)
        l_gm_loss = classification_loss + likelihood_loss
        return l_gm_loss, probability