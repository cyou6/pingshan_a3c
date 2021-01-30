from utils import *


class Policy:
    def __init__(self, n_s, n_a, n_w, n_f, n_step, n_fc_wave=128, n_fc_wait=32, n_fc_fp=32, n_lstm=64, name=None):
        self.name = 'Policy'
        if name is not None:
            self.name += '_' + str(name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step

        self.n_lstm = n_lstm
        self.n_fc_wave = n_fc_wave
        self.n_fc_wait = n_fc_wait
        self.n_fc_fp = n_fc_fp
        self.n_w = n_w
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s + n_w + n_f])  # forward 1-step
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s + n_w + n_f])  # backward n-step
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [2, n_lstm * 2])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi_fw, pi_state = self._build_net('forward', 'pi')
            self.v_fw, v_state = self._build_net('forward', 'v')
            pi_state = tf.expand_dims(pi_state, 0)
            v_state = tf.expand_dims(v_state, 0)
            self.new_states = tf.concat([pi_state, v_state], 0)
        with tf.variable_scope(self.name, reuse=True):
            self.pi, _ = self._build_net('backward', 'pi')
            self.v, _ = self._build_net('backward', 'v')
        self._reset()

    def _build_net(self, in_type, out_type):
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
        if out_type == 'pi':
            states = self.states[0]
        else:
            states = self.states[1]
        h0 = fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
        h1 = fc(ob[:, (self.n_s + self.n_w):], out_type + '_fcf', self.n_fc_fp)
        if self.n_w == 0:
            h = tf.concat([h0, h1], 1)
        else:
            h2 = fc(ob[:, self.n_s: (self.n_s + self.n_w)], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1, h2], 1)
        h, new_states = lstm(h, done, states, out_type + '_lstm')
        out_val = self._build_out_net(h, out_type)
        return out_val, new_states

    def _build_out_net(self, h, out_type):
        if out_type == 'pi':
            pi = fc(h, out_type, self.n_a, act=tf.nn.softmax)
            return tf.squeeze(pi)
        else:
            v = fc(h, out_type, 1, act=lambda x: x)
            return tf.squeeze(v)

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi)
        if 'v' in out_type:
            outs.append(self.v)
        return outs

    def _return_forward_outs(self, out_values):
        if len(out_values) == 1:
            return out_values[0]
        return out_values

    def _reset(self):
        # forget the cumulative states every cum_step
        self.states_fw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)
        self.states_bw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi_fw)
        if 'v' in out_type:
            outs.append(self.v_fw)
        return outs

    def forward(self, sess, ob, done, out_type='pv'):
        outs = self._get_forward_outs(out_type)
        # update state only when p is called
        if 'p' in out_type:
            outs.append(self.new_states)
        out_values = sess.run(outs, {self.ob_fw: np.array([ob]),
                                     self.done_fw: np.array([done]),
                                     self.states: self.states_fw})
        if 'p' in out_type:
            self.states_fw = out_values[-1]
            out_values = out_values[:-1]
        return self._return_forward_outs(out_values)

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                 summary_writer=None, global_step=None):
        if summary_writer is None:
            ops = self._train
        else:
            ops = [self.summary, self._train]
        outs = sess.run(ops,
                        {self.ob_bw: obs,
                         self.done_bw: dones,
                         self.states: self.states_bw,
                         self.A: acts,
                         self.ADV: Advs,
                         self.R: Rs,
                         self.lr: cur_lr,
                         self.entropy_coef: cur_beta})
        self.states_bw = np.copy(self.states_fw)
        if summary_writer is not None:
            summary_writer.add_summary(outs[0], global_step=global_step)

    def prepare_loss(self, v_coef, max_grad_norm, alpha, epsilon):
        self.A = tf.placeholder(tf.int32, [self.n_step])
        self.ADV = tf.placeholder(tf.float32, [self.n_step])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        self.entropy_coef = tf.placeholder(tf.float32, [])
        A_sparse = tf.one_hot(self.A, self.n_a)
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)
        entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
        policy_loss = -tf.reduce_mean(tf.reduce_sum(log_pi * A_sparse, axis=1) * self.ADV)
        value_loss = tf.reduce_mean(tf.square(self.R - self.v)) * 0.5 * v_coef
        self.loss = policy_loss + value_loss + entropy_loss

        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.lr = tf.placeholder(tf.float32, [])
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha,
                                                   epsilon=epsilon)
        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        # todo delete
        if self.name.endswith('_0a'):
            summaries = []
            # summaries.append(tf.summary.scalar('loss/%s_entropy_loss' % self.name, entropy_loss))
            # summaries.append(tf.summary.scalar('loss/%s_policy_loss' % self.name, policy_loss))
            summaries.append(tf.summary.scalar('loss/%s_value_loss' % self.name, value_loss))
            # summaries.append(tf.summary.scalar('loss/%s_total_loss' % self.name, self.loss))
            # summaries.append(tf.summary.scalar('train/%s_lr' % self.name, self.lr))
            # summaries.append(tf.summary.scalar('train/%s_entropy_beta' % self.name, self.entropy_coef))
            summaries.append(tf.summary.scalar('train/%s_gradnorm' % self.name, self.grad_norm))
            self.summary = tf.summary.merge(summaries)