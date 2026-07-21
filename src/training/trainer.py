"""
Trainer
========

Training utility for Two-Tower Recommendation Model.

TensorFlow 2.19
Keras 3
"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

import tensorflow as tf

from src.evaluation.retrieval import RetrievalEvaluator
from src.evaluation.metrics import (
    RecallAtK,
    PrecisionAtK,
    HitRateAtK,
    MRR,
    NDCGAtK,
)


class Trainer:
    """
    Generic trainer for the Two-Tower model.

    Responsibilities
    ----------------
    - Training loop
    - Validation loop
    - Checkpoint save/load
    - Retrieval evaluation
    """

    def __init__(
        self,
        model,
        retrieval_task,
        optimizer,
        callbacks=None,
    ):

        self.model = model
        self.retrieval_task = retrieval_task
        self.optimizer = optimizer

        self.history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        self.callbacks = callbacks or []
        self.stop_training = False
        

    # =====================================================
    # Training Step
    # =====================================================

    @tf.function
    def train_step(
        self,
        batch,
    ):

        user_features = {
            "user_id": batch["user_id"],
        }

        item_features = {
            "movie_id": batch["movie_id"],
        }

        with tf.GradientTape() as tape:

            output = self.model(
                {
                    "user": user_features,
                    "item": item_features,
                },
                training=True,
            )

            loss, metrics = self.retrieval_task(
                output.user_embedding,
                output.item_embedding,
            )

        gradients = tape.gradient(
            loss,
            self.model.trainable_variables,
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.model.trainable_variables,
            )
        )

        return loss, metrics

    # =====================================================
    # Validation Step
    # =====================================================

    @tf.function
    def validation_step(
        self,
        batch,
    ):

        user_features = {
            "user_id": batch["user_id"],
        }

        item_features = {
            "movie_id": batch["movie_id"],
        }

        output = self.model(
            {
                "user": user_features,
                "item": item_features,
            },
            training=False,
        )

        loss, metrics = self.retrieval_task(
            output.user_embedding,
            output.item_embedding,
        )

        return loss, metrics

        # =====================================================
    # Train Epoch
    # =====================================================

    def train_epoch(
        self,
        dataset,
    ):

        losses = []
        accuracies = []

        progress = tqdm(
            dataset,
            desc="Training",
            leave=False,
        )

        for batch in progress:

            loss, metrics = self.train_step(batch)

            loss_value = float(loss)
            acc_value = float(metrics["accuracy"])

            losses.append(loss_value)
            accuracies.append(acc_value)

            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                acc=f"{acc_value:.4f}",
            )

        return (
            sum(losses) / len(losses),
            sum(accuracies) / len(accuracies),
        )

    # =====================================================
    # Validation Epoch
    # =====================================================

    def validate_epoch(
        self,
        dataset,
    ):

        losses = []
        accuracies = []

        progress = tqdm(
            dataset,
            desc="Validation",
            leave=False,
        )

        for batch in progress:

            loss, metrics = self.validation_step(batch)

            loss_value = float(loss)
            acc_value = float(metrics["accuracy"])

            losses.append(loss_value)
            accuracies.append(acc_value)

            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                acc=f"{acc_value:.4f}",
            )

        return (
            sum(losses) / len(losses),
            sum(accuracies) / len(accuracies),
        )

    # =====================================================
    # Fit
    # =====================================================

    def fit(
        self,
        train_dataset,
        epochs,
        validation_dataset=None,
    ):

        print()
        print("=" * 60)
        print("Training Started")
        print("=" * 60)

        for cb in self.callbacks:
            cb.on_train_begin(self)

        for epoch in range(epochs):

            for cb in self.callbacks:
                cb.on_epoch_begin(self, epoch)

            print()
            print(f"Epoch {epoch + 1}/{epochs}")
            print("-" * 60)

            train_loss, train_acc = self.train_epoch(
                train_dataset,
            )

            self.history["loss"].append(train_loss)
            self.history["accuracy"].append(train_acc)

            print(
                f"Train Loss : {train_loss:.4f}"
            )
            print(
                f"Train Acc  : {train_acc:.4f}"
            )

            if validation_dataset is not None:

                val_loss, val_acc = self.validate_epoch(
                    validation_dataset,
                )

                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_acc)

                print(
                    f"Val Loss   : {val_loss:.4f}"
                )

                print(
                    f"Val Acc    : {val_acc:.4f}"
                )

                logs = {

                    "train_loss": train_loss,

                    "train_acc": train_acc,

                    "val_loss": val_loss,

                    "val_acc": val_acc,

                }
            
            for cb in self.callbacks:
                cb.on_epoch_end(
                    self,
                    epoch,
                    logs,
                )
            
            if self.stop_training:
                print("\nEarly stopping.")
                break

        print()
        print("=" * 60)
        print("Training Finished")
        print("=" * 60)

        for cb in self.callbacks:
            cb.on_train_end(self)

        return self.history

        # =====================================================
    # Save
    # =====================================================

    def save(
        self,
        path: str,
    ):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.save_weights(path)

        print()
        print(f"Model saved to: {path}")

    # =====================================================
    # Load
    # =====================================================

    def load(
        self,
        path: str,
    ):

        path = Path(path)

        self.model.load_weights(path)

        print()
        print(f"Model loaded from: {path}")

    # =====================================================
    # Evaluate
    # =====================================================

    def evaluate(
        self,
        test_dataset,
        item_dataset,
    ):

        evaluator = RetrievalEvaluator(
            model=self.model,
            item_dataset=item_dataset,
        )

        metrics = [

            RecallAtK(10),
            RecallAtK(20),

            PrecisionAtK(10),

            HitRateAtK(10),

            MRR(),

            NDCGAtK(10),

        ]

        print()
        print("=" * 60)
        print("Evaluation")
        print("=" * 60)

        results = evaluator.evaluate(
            test_dataset,
            metrics,
        )

        for name, value in results.items():

            print(
                f"{name:20s}: {value:.4f}"
            )

        return results

    # =====================================================
    # Summary
    # =====================================================

    def summary(self):

        self.model.summary()

        return self

    # =====================================================
    # Reset History
    # =====================================================

    def reset_history(self):

        self.history = {

            "loss": [],

            "accuracy": [],

            "val_loss": [],

            "val_accuracy": [],

        }

        return self