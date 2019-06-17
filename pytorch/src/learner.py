import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np



class Learner(nn.Module):
    """

    """

    def __init__(self, config, imgc, imgsz):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()


        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                self.func_conv2d(param)

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                self.func_batchnorm(param)


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            elif name is 'layer':
                indim = param[1]
                outdim = param[0]
                num_block = param[2]
                for j in range(num_block):
                    self.func_conv2d([outdim,indim,3,3])
                    self.func_batchnorm([outdim])
                    self.func_conv2d([outdim,indim,3,3])
                    self.func_batchnorm([outdim])

                    if indim != outdim:
                        self.func_conv2d([outdim,indim,1,1])
                        self.func_batchnorm([outdim])

                    indim = outdim
            else:
                raise NotImplementedError

    def func_conv2d(self,param):
        # [ch_out, ch_in, kernelsz, kernelsz]
        w = nn.Parameter(torch.ones(*param[:4]))
        # gain=1 according to cbfin's implementation
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)

        # [ch_out]
        self.vars.append(nn.Parameter(torch.zeros(param[0])))
    
    def op_conv2d(vars,idx,x,s,p):
        w, b = vars[idx], vars[idx + 1]
        # remember to keep synchrozied of forward_encoder and forward_decoder!
        x = F.conv2d(x, w, b, stride=s, padding=p)
        idx += 2
        return (x,idx)
        # print(name, param, '\tout:', x.shape)

    def func_batchnorm(self,param):
        # [ch_out]
        w = nn.Parameter(torch.ones(param[0]))
        self.vars.append(w)
        # [ch_out]
        self.vars.append(nn.Parameter(torch.zeros(param[0])))

        # must set requires_grad=False
        running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
        running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])

    def op_batchnorm(vars,idx,bn_idx,x,bn_training):
        w, b = vars[idx], vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
        x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
        idx += 2
        bn_idx += 2
        return (x,idx,bn_idx)

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'
            
            elif name is 'layer':
                tmp = 'resnet_layer%d:(block_num:%d)'%(param[3],param[2])
                info += tmp + '\n'                 

            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d)'%(param[0])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info



    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name is 'conv2d':
                (x,idx) = self.op_conv2d(vars,idx,x,param[4],param[5])
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                (x,idx,bn_idx) = self.op_batchnorm(vars,idx,bn_idx,x,bn_training)
            elif name is 'layer':
                indim = param[1]
                outdim = param[0]
                num_block = param[2]
                layer_id = param[3]
                
                for j in range(num_block):
                    half_res = (layer_id>=1) and (j==0)
                    ##c1
                    (out,idx) = self.op_conv2d(vars,idx,x,2 if half_res else 1,1)
                    ##bn1
                    (out,idx,bn_idx) = self.op_batchnorm(vars,idx,bn_idx,out,bn_training)
                    ##relu
                    out = F.relu(out, inplace=True)
                    ##c2
                    (out,idx) = self.op_conv2d(vars,idx,out,1,1)
                    ##bn2
                    (out,idx,bn_idx) = self.op_batchnorm(vars,idx,bn_idx,out,bn_training)
                    ##shortcut
                    if indim!=outdim:
                        (x,idx) = self.op_conv2d(vars,idx,x,2 if half_res else 1,0)
                        (x,idx,bn_idx) = self.op_batchnorm(vars,idx,bn_idx,x,bn_training)
                    out = out + x
                    ##relu
                    x = F.relu(out, inplace=True)
                    indim = outdim
            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)


        return x


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars