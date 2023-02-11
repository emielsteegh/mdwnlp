# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace NLP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Open Data Rechtspraak dutch topic classification dataset."""

from __future__ import absolute_import, division, print_function

import csv
import os

import nlp


_DESCRIPTION = """\
still a WIP, Dataset originally comes from Open Data van de Rechtspraak"
"""

_TRAIN_DOWNLOAD_URL = (
    "https://huggingface.co/datasets/Rodekool/ornl8/resolve/main/train.csv"
)
_TEST_DOWNLOAD_URL = (
    "https://huggingface.co/datasets/Rodekool/ornl8/resolve/main/test.csv"
)


class ORnl(nlp.GeneratorBasedBuilder):
    """Open Data van de Rechtspraak dutch topic classification dataset."""

    def _info(self):
        return nlp.DatasetInfo(
            description=_DESCRIPTION,
            features=nlp.Features(
                {
                    "text": nlp.Value("string"),
                    "label": nlp.features.ClassLabel(
                        names=[
                            "Ambtenarenrecht",
                            "Arbeidsrecht",
                            "Belastingrecht",
                            "Omgevingsrecht",
                            "Personen- en familierecht",
                            "Socialezekerheidsrecht",
                            "Verbintenissenrecht",
                            "Vreemdelingenrecht",
                        ]
                    ),
                }
            ),
            homepage="https://www.rechtspraak.nl/Uitspraken/paginas/open-data.aspx",
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            nlp.SplitGenerator(
                name=nlp.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            nlp.SplitGenerator(name=nlp.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate Rechtspraak examples."""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file,
                quotechar='"',
                delimiter=",",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True,
            )
            for id_, row in enumerate(csv_reader):
                label, title, description = row

                # Original labels are [1, 2, 3, 4, 5, 6, 7, 8] ->
                #                   ['Ambtenarenrecht',
                #                    'Arbeidsrecht',
                #                    'Belastingrecht',
                #                    'Omgevingsrecht',
                #                    'Personen- en familierecht',
                #                    'Socialezekerheidsrecht',
                #                    'Verbintenissenrecht',
                #                    'Vreemdelingenrecht']
                # Re-map to [0, 1, 2, 3, 4, 5, 6, 7].

                label = int(label) - 1
                text = " ".join((title, description))
                yield id_, {"text": text, "label": label}
