#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 09:09:17 2019

@author: tony
"""



""""
https://github.com/zhengyang-wang/3D-Unet--Tensorflow/blob/master/utils/attention.py
This script defines 3D different multi-head attention layers.

"""
#attention for neurites

import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiHeadAttention(nn.Module):
    def __init__(self,
                 chanIn,
                 output_filters,
                 total_key_filters,
                 total_value_filters,                 
                 num_heads,
                 layer_type='SAME'):
        super(MultiHeadAttention, self).__init__()
        
        '''
        inputs: channels first as input [batch, chn, h,w]
        
        
        '''
        
        """3d Multihead scaled-dot-product attention with input/output transformations.

        Args:\n"
            inputs: a Tensor with shape [batch, h, w, channels]
            
            total_key_filters: an integer. Note that queries have the same number 
                of channels as keys
            total_value_filters: an integer
            output_depth: an integer
            num_heads: an integer dividing total_key_filters and total_value_filters
            layer_type: a string, type of this layer -- SAME, DOWN, UP
            name: an optional string
        Returns:
            A Tensor of shape [batch, _d, _h, _w, output_filters]

        Raises:
            ValueError: if the total_key_filters or total_value_filters are not divisible
                by the number of attention heads.
        """

        if total_key_filters % num_heads != 0:
            raise ValueError("Key depth (%d) must be divisible by the number of "
                            "attention heads (%d)." % (total_key_filters, num_heads))
        if total_value_filters % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the number of "
                            "attention heads (%d)." % (total_value_filters, num_heads))
        if layer_type not in ['SAME', 'DOWN', 'UP']:
            raise ValueError("Layer type (%s) must be one of SAME, "
                            "DOWN, UP." % (layer_type))
            
        
        '''
        inputs: [batch, chn, d, h,w]
        output: [batch, chn, d, h,w]
        next step, permute to [batch, d,h,w,chn]
        '''
        self.q = compute_qkv(chanIn, total_key_filters,'q',layer_type=layer_type)
        self.k = compute_qkv(chanIn, total_key_filters,'k',layer_type=layer_type)
        self.v = compute_qkv(chanIn, total_value_filters,'v',layer_type=layer_type)
        
        self.num_heads = num_heads
        self.total_key_filters = total_key_filters
        self.total_value_filters = total_value_filters
        
        self.conv = nn.Conv3d(total_key_filters, output_filters, 1,1,bias=True)
        
    def forward(self,x):
        
#        print('x before q shape in: ', x.shape)
        #[batch, chn, h,w]
        q = self.q(x)
#        print('q shape in: ', q.shape)
        k = self.k(x)
#         print('k shape in: ', k.shape)
        v = self.v(x)
#         print('v shape in: ', v.shape)
        
        #permute to set [batch,d,h,w, chn]
        q = q.permute(0,2,3,4,1)
#        print('q after permute ', q.shape)
        k = k.permute(0,2,3,4,1)
        v = v.permute(0,2,3,4,1)
        
#        print('q shape before split ' ,q.shape[3], q.shape)
        q = split_heads(q,self.num_heads)    
#        print('q shape after split ' ,q.shape[3], q.shape)
        k = split_heads(k,self.num_heads)
        v = split_heads(v,self.num_heads)
        #after split [batch, heads, d,h,w,k/h]
#         print('q split ', q.shape)
       
        #normalize
        key_filters_per_head = self.total_key_filters // self.num_heads
        q *= key_filters_per_head**-0.5
        
        att = global_attention(q,k,v, q.shape[2])
#        print('out of attt : ', att.shape)



        x = att #combine_heads(att)
        
#         print('LAST shape in, ' ,x.shape)
#        x = x.permute(0,3,1,2)
#         print('LAST shape in, ' ,x.shape)
        x = self.conv(x)
#        print('out of att', x.shape)
        return x



def compute_qkv(chanIn,filters, qkv,layer_type='SAME'):
    '''
    Args:
        inputs: a Tensor with shape [batch,channels, h, w]
        total_key_filters: an integer
        total_value_filters: and integer
        layer_type: String, type of this layer -- SAME, DOWN, UP

    Returns:
        q: [batch, _h, _w, total_key_filters] tensor
        k: [batch, h, w, total_key_filters] tensor
		v: [batch, h, w, total_value_filters] tensor

    
    
    '''
#     print('compute_qkv .........')
    if qkv == 'q':
        # linear transformation for q , filters = key_filters
        if layer_type == 'SAME':
            qkv = nn.Conv3d(chanIn, filters, 1, 1, bias=True,padding=0)
        elif layer_type == 'DOWN':
            qkv = nn.Conv3d(chanIn, filters, 3, 2, bias=True, padding =1)
        elif layer_type == 'UP':
            qkv = nn.ConvTranspose3d(chanIn, filters, 3, 2, bias=True, padding=1,output_padding=1)
    
    if qkv == 'k':
        # linear transformation for k
        qkv = nn.Conv3d(chanIn, filters, 1, 1, bias=True, padding=0)

    if qkv =='v':
        # linear transformation for k, value filtesr
        qkv = nn.Conv3d(chanIn, filters, 1, 1, bias=True, padding=0)

    return qkv


def split_heads(x, num_heads):
    """Split channels (last dimension) into multiple heads (becomes dimension 1).

    Args:
        x: a Tensor with shape [batch, h, w, channels]
        num_heads: an integer

    Returns:
        a Tensor with shape [batch, num_heads, h, w, channels / num_heads]
    """
    
    return split_last_dimension(x, num_heads).permute(0,5,1,2,3,4)#permute(0,3,1,2,4)

def split_last_dimension(x,n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
        x: a Tensor with shape [..., m]
        n: an integer.
    Returns:
        a Tensor with shape [..., n, m/n]
    """
    

    chunk_size = int(x.shape[4]/ n)
    ret = torch.unsqueeze(x,5)
    
    ret = torch.cat(ret.split(split_size=chunk_size, dim=4),5)#.permute(0,1,2,4,3)
#    print('split', ret.shape)
#     ret.view(new_shape)
#     print('split view ', ret.shape)
    return ret





####################################################################################################

def global_attention(q, k, v, chan_z):
    """global self-attention.
    Args:
        q: a Tensor with shape [batch, heads, _d, _h, _w, channels_k]
        k: a Tensor with shape [batch, heads, d, h, w, channels_k]
        v: a Tensor with shape [batch, heads, d, h, w, channels_v]
        name: an optional string
    Returns:
        a Tensor of shape [batch, heads, _d, _h, _w, channels_v]
    """
#    print('shape q:', q.shape)
#    print('global att, v shape', v.shape, v.shape[-1])
#     print(new_shape)
    # flatten q,k,v
    q_new = flatten(q)
    k_new = flatten(k)
    v_new = flatten(v)
    
#    print('shape q after flat:', q_new.shape)
    

    # attention
    output = dot_product_attention(q_new, k_new, v_new, bias=None,
                dropout_rate=0.5)
#    print('outp ', output.shape)

    # putting the representations back in the right place
#     print('output before scatter', output.shape)
    output = scatter(output, chan_z)

    return output

def dot_product_attention(q, k, v, bias,dropout_rate=0.0):
    """Dot-product attention.
    Args:
        q: a Tensor with shape [batch, heads, length_q, channels_k]
        k: a Tensor with shape [batch, heads, length_kv, channels_k]
        v: a Tensor with shape [batch, heads, length_kv, channels_v]
        bias: bias Tensor
        dropout_rate: a floating point number
        name: an optional string
    Returns:
        A Tensor with shape [batch, heads, length_q, channels_v]
    """

    

    # [batch, num_heads, length_q, length_kv]
#     print(q.shape, k.transpose(2,3).shape)
    logits = torch.matmul(q, k.transpose(2,3))

    if bias is not None:
        logits += bias

    weights = F.softmax(logits)

    # dropping out the attention links for each of the heads
    weights = F.dropout(weights, dropout_rate)

    return torch.matmul(weights, v)


#def reshape_range(tensor, i, j, shape):
#    """Reshapes a tensor between dimensions i and j."""
#
#    target_shape = tf.concat(
#            [tf.shape(tensor)[:i], shape, tf.shape(tensor)[j:]],
#            axis=0)
#
#    return tf.reshape(tensor, target_shape)



def scatter(x, chn):
    """scatter x."""
    #combine heads and m/n
    x = x.view(x.shape[0],x.shape[1]*x.shape[3],-1)
#    print('scatter ', x.shape)
    x = x.view(x.shape[0],x.shape[1],chn,chn,-1)

    return x


def flatten(x):
    """flatten x."""
    # [batch, heads, h,w, ch/h]
    # [batch, heads, length, channels], length = d*h*w
    
    l  = x.shape[2] * x.shape[3]*x.shape[4]
    # [batch, heads, length, channels], length = d*h*w
    x = x.view(x.shape[0], x.shape[1], l,-1)
#     print('flatten shape', x.shape)

    return x


def combine_heads(x):
    """Inverse of split_heads_3d.
    Args:
        x: a Tensor with shape [batch, num_heads, d, h, w, channels / num_heads]
    Returns:
        a Tensor with shape [batch, d, h, w, channels]
    """
#  [0, 2, 3, 4, 1, 5]
#     print('combine heads ,', x.shape)
    return combine_last_two_dimensions(x)


def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
    Args:
        x: a Tensor with shape [..., a, b]
    Returns:
        a Tensor with shape [..., a*b]
    """

#     old_shape = x.get_shape().dims
#     a, b = old_shape[-2:]
#     new_shape = old_shape[:-2] + [a * b if a and b else None]

#     ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
#     ret.set_shape(new_shape)
    
    x = x.contiguous().view(x.shape[0],x.shape[1], x.shape[2], -1)
#     print('combine last two ', x.shape)

    return x