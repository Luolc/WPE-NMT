"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

from __future__ import division

import torch.nn as nn

import onmt.inputters as inputters
import onmt.utils

from onmt.utils.logging import logger


def build_trainer(opt, model, fields, optim, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    report_manager = onmt.utils.build_report_manager(opt)
    trainer = Trainer(model, fields, optim, report_manager, model_saver=model_saver)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, fields, optim, report_manager=None, model_saver=None):
        # Basic attributes.
        self.model = model
        self.fields = fields
        self.optim = optim
        self.report_manager = report_manager
        self.model_saver = model_saver

        self.creterion = nn.NLLLoss(size_average=False)  # todo: weight

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter_fct, valid_iter_fct, train_steps, valid_steps):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('Start training...')

        step = self.optim._step + 1
        train_iter = train_iter_fct()

        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:
            for i, batch in enumerate(train_iter):
                normalization = batch.batch_size
                self.step(batch, normalization, total_stats, report_stats)

                report_stats = self._maybe_report_training(
                    step, train_steps,
                    self.optim.learning_rate,
                    report_stats)

                if step % valid_steps == 0:
                    valid_iter = valid_iter_fct()
                    valid_stats = self.validate(valid_iter)
                    valid_stats = self._maybe_gather_stats(valid_stats)
                    self._report_step(self.optim.learning_rate,
                                      step, valid_stats=valid_stats)

                self._maybe_save(step)
                step += 1
                if step > train_steps:
                    break
            train_iter = train_iter_fct()

        return total_stats

    def step(self, batch, normalization, total_stats, report_stats):
        dec_state = None
        src = inputters.make_features(batch, 'src', 'text')
        _, src_lengths = batch.src
        report_stats.n_src_words += src_lengths.sum().item()

        tgt = inputters.make_features(batch, 'tgt')

        # 2. F-prop all AND generator.
        self.model.zero_grad()
        outputs, attns, dec_state = self.model(src, tgt, src_lengths, dec_state)
        scores = self.model.generator(outputs.view(-1, outputs.size(2)))

        # 3. Compute loss in shards for memory efficiency.
        target = batch.tgt[1:batch.tgt.size(0)]
        gtruth = target.view(-1)
        loss = self.creterion(scores, gtruth)
        loss.div(float(normalization)).backward()
        batch_stats = self._stats(loss.data.clone(), scores.data, target.view(-1).data)

        total_stats.update(batch_stats)
        report_stats.update(batch_stats)

        self.optim.step()

    def _stats(self, loss, scores, target):
        padding_idx = self.fields["tgt"].vocab.stoi[inputters.PAD_WORD]
        pred = scores.argmax(1)
        non_padding = target.ne(padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct)

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = onmt.utils.Statistics()

        for batch in valid_iter:
            src = inputters.make_features(batch, 'src', 'text')
            _, src_lengths = batch.src
            tgt = inputters.make_features(batch, 'tgt')

            # F-prop through the model.
            outputs, attns, _ = self.model(src, tgt, src_lengths)

            # Compute loss.
            scores = self.model.generator(outputs.view(-1, outputs.size(2)))
            target = batch.tgt[1:batch.tgt.size(0)]
            gtruth = target.view(-1)
            loss = self.creterion(scores, gtruth)
            batch_stats = self._stats(loss.data.clone(), scores.data, target.view(-1).data)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return onmt.utils.Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
