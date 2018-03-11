import numpy as np
import tensorflow as tf

from dfp import tf_ops as my_ops
from dfp.agent import Agent


class AgentMultimodalAdvantage(Agent):
    def make_net(self, input_sensory, input_actions, input_objectives, reuse=False):
        """
        Hooks up network for inferring non-observed modalities for given time step and for
           predicting future targets (measurement + subset of modalities)

        Args:
            input_sensory - tf placeholder for all modalities (includes both observed and ground truth for to be inferred modalities)
            input_actions - tf placeholder for one hot vector representation of action taken by agent
            input_objectives - tf placeholder for objective coefficients
                (objective is a linear weighted function of predicted measurements, these are the weights specified by the user)
        Returns:
            infer_sensory_embeddings - dictionary with computed embeddings for infer_modalities
            pred_all - All future target predictions (for all actions)
            pred_relevant - Relevant future target prediction (for current action)
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()
        if self.random_objective_coeffs:
            assert isinstance(self.obj_fc_params, np.ndarray), 'Need fc_obj_params with randomized objectives'

        self.fc_val_params = np.copy(self.joint_fc_params)
        self.fc_val_params['out_dims'][-1] = self.target_dim
        self.fc_adv_params = np.copy(self.joint_fc_params)
        self.fc_adv_params['out_dims'][-1] = len(self.net_discrete_actions) * self.target_dim

        sensory_embeddings = {}

        # build nets to embed input_modalities into concatenated representation
        for modality in self.input_modalities:
            if modality == 'color':
                img_conv = my_ops.conv_encoder(input_sensory['color'], self.img_conv_params, 'img_conv', msra_coeff=0.9)
                img_fc = my_ops.fc_net(my_ops.flatten(img_conv), self.img_fc_params, 'img_fc', msra_coeff=0.9)
                sensory_embeddings['color'] = img_fc
            elif modality == 'depth':
                depth_conv = my_ops.conv_encoder(input_sensory['depth'], self.depth_conv_params, 'depth_conv', msra_coeff=0.9)
                depth_fc = my_ops.fc_net(my_ops.flatten(depth_conv), self.depth_fc_params, 'depth_fc', msra_coeff=0.9)
                sensory_embeddings['depth'] = depth_fc
            elif modality == 'measurements':
                meas_fc = my_ops.fc_net(input_sensory['measurements'], self.meas_fc_params, 'meas_fc', msra_coeff=0.9)
                sensory_embeddings['measurements'] = meas_fc
            elif modality == 'force':
                force_fc = my_ops.fc_net(input_sensory['force'], self.force_fc_params, 'force_fc', msra_coeff=0.9)
                sensory_embeddings['force'] = force_fc
            elif modality == 'audiopath':
                audiopath_fc = my_ops.fc_net(input_sensory['audiopath'], self.audiopath_fc_params, 'audiopath_fc', msra_coeff=0.9)
                sensory_embeddings['audiopath'] = audiopath_fc
            elif modality == 'audio':
                audio_fc = my_ops.fc_net(input_sensory['audio'], self.audio_fc_params, 'audio_fc', msra_coeff=0.9)
                sensory_embeddings['audio'] = audio_fc
            elif modality == 'actions':
                actions_fc = my_ops.fc_net(input_sensory['actions'], self.actions_fc_params, 'actions_fc', msra_coeff=0.9)
                sensory_embeddings['actions'] = actions_fc
            elif modality == 'goalRoomType':
                goal_roomtype_fc = my_ops.fc_net(input_sensory['goalRoomType'], self.goalroomtype_fc_params, 'goalroomtype_fc', msra_coeff=0.9)
                sensory_embeddings['goalRoomType'] = goal_roomtype_fc
            elif modality == 'roomType':
                roomtype_fc = my_ops.fc_net(input_sensory['roomType'], self.roomtype_fc_params, 'roomtype_fc', msra_coeff=0.9)
                sensory_embeddings['roomType'] = roomtype_fc
            else:
                raise Exception('Unsupported input modality %s' % modality)

        # is there a better way to get values from a dictionary ordered by key?
        input_concat_fc = tf.concat([sensory_embeddings[modality] for modality in sorted(sensory_embeddings)], 1)

        # infer modalities from input_concat_fc
        infer_sensory_embeddings = {}
        for modality in self.infer_modalities:
            if modality == 'measurements':
                # handle this one below
                pass
            elif modality == 'roomType':
                self.infer_roomtype_fc_params['out_dims'][-1] = input_sensory['roomType'].get_shape().as_list()[1]
                roomtype_fc = my_ops.fc_net_with_soft_max(input_concat_fc, self.infer_roomtype_fc_params, 'infer_roomType_fc', msra_coeff=0.9)
                sensory_embeddings[modality] = roomtype_fc
                infer_sensory_embeddings[modality] = roomtype_fc
            else:
                raise Exception('Unsupported infer modality %s' % modality)

        if 'measurements' in self.infer_modalities:
            input_inferred_concat_fc = tf.concat([sensory_embeddings[modality] for modality in sorted(sensory_embeddings)], 1)
            self.infer_meas_fc_params['out_dims'][-1] = input_sensory['measurements'].get_shape().as_list()[1]
            meas_fc = my_ops.fc_net(input_inferred_concat_fc, self.infer_meas_fc_params, 'infer_meas_fc', msra_coeff=0.9)
            sensory_embeddings['measurements'] = meas_fc
            infer_sensory_embeddings['measurements'] = meas_fc

        # add objectives to embedding
        if isinstance(self.obj_fc_params, np.ndarray):
            obj_fc = my_ops.fc_net(input_objectives, self.obj_fc_params, 'obj_fc', msra_coeff=0.9)
            sensory_embeddings['objectives'] = obj_fc

        # final input + inferred concatenated
        concat_fc = tf.concat([sensory_embeddings[modality] for modality in sorted(sensory_embeddings)], 1)

        # predicted expectation over all actions
        pred_val_fc = my_ops.fc_net(concat_fc, self.fc_val_params, 'pred_val_fc', last_linear=True, msra_coeff=0.9)
        # predicted action-conditional differences
        pred_adv_fc = my_ops.fc_net(concat_fc, self.fc_adv_params, 'pred_adv_fc', last_linear=True, msra_coeff=0.9)

        adv_reshape = tf.reshape(pred_adv_fc, [-1, len(self.net_discrete_actions), self.target_dim])

        pred_all_nomean = adv_reshape - tf.reduce_mean(adv_reshape, reduction_indices=1, keep_dims=True)
        pred_all = pred_all_nomean + tf.reshape(pred_val_fc, [-1, 1, self.target_dim])
        pred_relevant = tf.boolean_mask(pred_all, tf.cast(input_actions, tf.bool))

        return infer_sensory_embeddings, pred_all, pred_relevant

    def make_losses(self, infer_sensory, input_infer_sensory_preprocessed,
                    pred_relevant, targets_preprocessed, objective_indices, objective_coeffs):
        """
        Setup the losses

        Args:
            infer_sensory - dictionary of inferred modalities
            input_infer_sensory_preprocessed - dictionary of actual ground truth modalities
            pred_relevant - future target predictions that are relevant to the current action
            targets_preprocessed - ground truth of future targets
            objective_indices
            objective_coeffs
        Returns:
            full_loss
            errs_to_print
            short_summary
            detailed_summary
        """

        # make a loss function and compute some summary numbers

        # Future prediction target losses
        per_target_loss = my_ops.mse_ignore_nans(pred_relevant, targets_preprocessed, reduction_indices=0)
        target_loss = tf.reduce_sum(per_target_loss)

        # Inferred modality losses
        per_infer_sensory_loss = []
        for modality in self.infer_modalities:
            # TODO: crossentropy loss for roomType (one per history element)
            sensory_loss = my_ops.mse_ignore_nans(infer_sensory[modality],
                                                  input_infer_sensory_preprocessed[modality],
                                                  reduction_indices=0)
            per_infer_sensory_loss.append(tf.reduce_sum(sensory_loss))
        infer_sensory_loss = tf.reduce_sum(per_infer_sensory_loss)

        # Combined loss
        loss = target_loss + infer_sensory_loss

        # compute objective value, just for logging purposes
        #print(objective_coeffs[None,:].shape, targets_preprocessed[:,objective_indices].get_shape())
        # TODO add multiplication by the objective_coeffs (somehow not trivial)
        obj = tf.reduce_sum(self.postprocess_predictions(targets_preprocessed), 1)
        #obj = tf.sum(self.postprocess_predictions(targets_preprocessed[:,objective_indices]) * objective_coeffs[None,:], axis=1)
        obj_nonan = tf.where(tf.is_nan(obj), tf.zeros_like(obj), obj)
        num_valid_targets = tf.reduce_sum(1-tf.cast(tf.is_nan(obj), tf.float32))
        mean_obj = tf.reduce_sum(obj_nonan) / num_valid_targets

        # summaries
        obj_sum = tf.summary.scalar("objective", mean_obj)
        #TODO
        per_target_loss_sums = []
        #per_target_loss_sums = [tf.summary.scalar(name, loss) for name,loss in zip(self.target_names,per_target_loss)]
        loss_infer_sum = tf.summary.scalar("infer_loss", infer_sensory_loss)
        loss_sum = tf.summary.scalar("full_loss", loss)

        #self.per_target_loss = tf.get_variable('avg_targets', [self.target_dim], initializer=tf.constant_initializer(value=0.))

        full_loss = loss
        errs_to_print = [loss]
        short_summary = [loss_sum, loss_infer_sum]
        detailed_summary = per_target_loss_sums + [obj_sum]

        return full_loss, errs_to_print, short_summary, detailed_summary

    def act_net(self, states, objective_coeffs):
        """
        Select action given a state and objective_coeffs

        Args:
            states
            objective_coeffs
        Returns:
            action
        """

        if objective_coeffs.ndim == 1:
            curr_objective_coeffs = np.tile(objective_coeffs[None,:],(states['measurements'].shape[0],1))
        else:
            curr_objective_coeffs = objective_coeffs

        feed_dict = {self.input_sensory[m]: states[m] for m in self.input_modalities}
        feed_dict.update({self.input_objective_coeffs: curr_objective_coeffs})
        predictions = self.sess.run(self.pred_all, feed_dict=feed_dict)

        self.curr_predictions = predictions[:,:,self.objective_indices]*curr_objective_coeffs[:,None,:]
        self.curr_objectives = np.sum(self.curr_predictions, axis=2)
        #print(predictions[:,:,self.objective_params[0]])
        #print(curr_objective)
        #print(objectives)
        #print(np.argmax(objectives, axis=1))

        curr_action = np.argmax(self.curr_objectives, axis=1)
        #self.previous_actions = curr_action
        return curr_action
