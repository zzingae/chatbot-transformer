import tensorflow as tf
from official.transformer.model.transformer import Transformer

from official.transformer.model import attention_layer
from official.transformer.model import beam_search
from official.transformer.model import embedding_layer
from official.transformer.model import ffn_layer
from official.transformer.model import model_utils
from official.transformer.utils.tokenizer import EOS_ID


def make_mask(prob, i, L, T):

    batch_size = tf.shape(prob)[0]
    # number of masking decreases as i increases
    mask_num = tf.cast((T-i)/T * L, tf.int32)
    # find smallest k probabilities indices for each batch (-prob for ascending order)
    _, min_prob_inds = tf.math.top_k(-prob, k=mask_num, sorted=True, name=None)

    # a=[0,1,..batch_size], a.shape=(batch_size,1)
    # b=[1,1,..,1], b.shape=(1,mask_num)
    # c=[[0,..,0][1,..,1][2,..,2]], c.shape=(batch_size,mask_num)
    a=tf.expand_dims(tf.range(0,batch_size,dtype=tf.int32),1)
    b=tf.expand_dims(tf.ones(mask_num, dtype=tf.int32),0)
    c=a*b

    # stack[c,min_prob_inds] makes 2d indices (r,c), where masking 1s are inserted
    indices = tf.reshape(tf.stack([c,min_prob_inds],axis=2), [batch_size*mask_num,2])
    mask = tf.scatter_nd(indices, tf.ones(batch_size*mask_num,tf.int32), tf.shape(prob))

    return tf.cast(mask, tf.bool)

def get_decoder_self_attention_bias(length):
    """Calculate bias for decoder that maintains model's autoregressive property.

    Creates a tensor that masks out locations that correspond to illegal
    connections, so prediction at position i cannot draw information from future
    positions.

    Args:
    length: int length of sequences in batch.

    Returns:
    float tensor of shape [1, 1, length, length]
    """
    with tf.name_scope("decoder_self_attention_bias"):
        # valid_locs = tf.linalg.band_part(tf.ones([length, length]), -1, 0)
        valid_locs = tf.ones([length, length])
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        
        decoder_bias = _NEG_INF * (1.0 - valid_locs)
    return decoder_bias



class MaskTransformer(Transformer):

  def decode(self, targets, encoder_outputs, attention_bias):
    """Generate logits for each value in the target sequence.

    Args:
      targets: target values for the output sequence.
        int tensor with shape [batch_size, target_length]
      encoder_outputs: continuous representation of input sequence.
        float tensor with shape [batch_size, input_length, hidden_size]
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length]

    Returns:
      float32 tensor with shape [batch_size, target_length, vocab_size]
    """
    with tf.name_scope("decode"):
      # Prepare inputs to decoder layers by shifting targets, adding positional
      # encoding and applying dropout.
      decoder_inputs = self.embedding_softmax_layer(targets)

      # removed shifting targets
      # with tf.name_scope("shift_targets"):
      #   # Shift targets to the right, and remove the last element
      #   decoder_inputs = tf.pad(
      #       decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
      with tf.name_scope("add_pos_encoding"):
        length = tf.shape(decoder_inputs)[1]
        decoder_inputs += model_utils.get_position_encoding(
            length, self.params["hidden_size"])
      if self.train:
        decoder_inputs = tf.nn.dropout(
            decoder_inputs, 1 - self.params["layer_postprocess_dropout"])

      # removed attention masking
    #   decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(length)
      decoder_self_attention_bias = get_decoder_self_attention_bias(length)
      outputs = self.decoder_stack(
          decoder_inputs, encoder_outputs, decoder_self_attention_bias,
          attention_bias)
      logits = self.embedding_softmax_layer.linear(outputs)
      return logits

    def predict(self, y, p, i, encoder_outputs, attention_bias):
        # conditional masked prediction with iterations
        mask = tf.cond(tf.math.equal(i,0), 
                        # starting with all masking
                        true_fn=lambda: tf.ones([self.params['batch_size'], self.params['max_decode_length']],tf.bool),
                        false_fn=lambda: make_mask(p,i,self.params['max_decode_length'],self.params['max_iter']))

        y_mask = tf.where(mask, tf.zeros(tf.shape(y),tf.int32), y)
        logits = self.decode(y_mask, encoder_outputs, attention_bias)
        # logits shape: [batch_size, target_length, vocab_size]
        y_new = tf.math.argmax(logits, axis=2,output_type=tf.int32)
        y_new = tf.where(mask, y_new, y)
        p_new = tf.math.reduce_max(logits, axis=2)
        p_new = tf.where(mask, p_new, p)

        i = i+1
        return y_new, p_new, i

    def __call__(self, inputs, targets=None):
        """Calculate target logits or inferred target sequences.

        Args:
        inputs: int tensor with shape [batch_size, input_length].
        targets: None or int tensor with shape [batch_size, target_length].

        Returns:
        If targets is defined, then return logits for each word in the target
        sequence. float tensor with shape [batch_size, target_length, vocab_size]
        If target is none, then generate output sequence one token at a time.
        returns a dictionary {
            output: [batch_size, decoded length]
            score: [batch_size, float]}
        """
        # Variance scaling is used here because it seems to work in many problems.
        # Other reasonable initializers may also work just as well.
        initializer = tf.variance_scaling_initializer(
            self.params["initializer_gain"], mode="fan_avg", distribution="uniform")
        with tf.variable_scope("Transformer", initializer=initializer):
            # Calculate attention bias for encoder self-attention and decoder
            # multi-headed attention layers.

            # TODO change this to "model_utils.get_padding_bias(inputs)" // flrngel
            attention_bias = tf.zeros((self.params['batch_size'], 1, 1, inputs.shape[1]))
            # Run the inputs through the encoder layer to map the symbol
            # representations to continuous representations.
            encoder_outputs = self.encode(inputs, attention_bias)

            # Generate output sequence if targets is None, or return logits if target
            # sequence is known.
            if targets is None:
                y=tf.zeros([self.params['batch_size'], self.params['max_decode_length']],tf.int32)
                p=tf.zeros([self.params['batch_size'], self.params['max_decode_length']],tf.float32)
                i=0

                cond = lambda y, p, i: tf.less(i, self.params['max_iter'])
                body = lambda y, p, i: self.predict(y, p, i, encoder_outputs, attention_bias)

                final_y, _, _ = tf.while_loop(cond, body, [y, p, i])
                return final_y
                # return self.predict(encoder_outputs, attention_bias)
            else:
                logits = self.decode(targets, encoder_outputs, attention_bias)
                return logits

