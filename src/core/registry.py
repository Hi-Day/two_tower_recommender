"""
Generic Registry

Reusable registry for
encoders,
combiners,
projection heads,
losses,
samplers,
retrievers.
"""

from __future__ import annotations


class Registry:

    def __init__(self):

        self._objects = {}

    # --------------------------------------------

    def register(self, name):

        def wrapper(cls):

            key = name.lower()

            if key in self._objects:

                raise ValueError(

                    f"{key} already exists."

                )

            self._objects[key] = cls

            return cls

        return wrapper

    # --------------------------------------------

    def build(self, name, *args, **kwargs):

        key = name.lower()

        if key not in self._objects:

            raise ValueError(

                f"{key} not registered."

            )

        return self._objects[key](

            *args,

            **kwargs,

        )

    # --------------------------------------------

    def available(self):

        return sorted(

            self._objects.keys()

        )