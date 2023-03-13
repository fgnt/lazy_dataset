"""
Commands to inspect a database json.

Example calls:

    $ python -m lazy_dataset.database_cli preview <jsonpath>
    $ python -m lazy_dataset.database_cli preview <jsonpath> 1 3
    $ python -m lazy_dataset.database_cli diff <jsonpath1> <jsonpath2>

"""

import sys
import contextlib
import shutil
from pathlib import Path

import IPython.lib.pretty

import paderbox as pb




class _c:  # noqa
    Color_Off = '\033[0m'  # Text Reset
    Black = '\033[0;30m'  # Black
    Red = '\033[0;31m'  # Red
    Green = '\033[0;32m'  # Green
    Yellow = '\033[0;33m'  # Yellow
    Blue = '\033[0;34m'  # Blue
    Purple = '\033[0;35m'  # Purple
    Cyan = '\033[0;36m'  # Cyan
    White = '\033[0;37m'  # White
    Gray = '\033[0;90m'  # White


def preview(
        json, n=1, d=10, s=4, max_width=None,
        color=__name__ == '__main__',
):
    """
    Print a preview of a json, first `n` examples for the first `d` datasets in
    a json are printed.

    ToDo:
        diff option.
        icdiff <(python -m padercontrib.database.show_json wham_mix_both_max_8k.json)

    Args:
        json: Path to a database json
        n: Number of examples that will be printed.
        d: Number of datasets that will be printed. In contrast to `n`, the
            dataset name will be printed, but the content removed.
        s: (Option max_seq_length from IPython) Control how many entries
            of a sequence are printed.
            Note: A value of 0 will print all values, see
                  `IPython.lib.pretty.pprint`.
        max_width: Tried maximum with of the output, default terminal width
            or 79.
        color: Bool, whether to use color codes.

    Returns:

        >> preview(json='/net/vol/jenkins/jsons/wsj.json', n=1, d=1, color=False)
        {
            'datasets': {
                'cv_dev93': {
                    '4k0c0301': {
                        'audio_path': {'observation': '/net/fastdb/wsj/....wav'},
                        'example_id': '...',
                        'gender': 'male',
                        'kaldi_transcription': '...',
                        'num_samples': {'observation': 207299},
                        'speaker_id': '...',
                        'transcription': '...',
                    },
                    ...,  # skipped 502 of 503 examples
                },
                'cv_dev93_5k': ...,  # 513 examples
                'test_eval92': ...,  # 333 examples
                'test_eval92_5k': ...,  # 330 examples
                'test_eval93': ...,  # 213 examples
                'test_eval93_5k': ...,  # 215 examples
                'train_si284': ...,  # 37416 examples
                'train_si84': ...,  # 7138 examples
            },
        }

    """
    if max_width is None:
        max_width = shutil.get_terminal_size((79, 20)).columns

    if Path(json).exists():
        data = pb.io.load(json)
    elif '+' in str(json):
        json = str(json)
        jsons = json.split('+')
        data = pb.utils.nested.nested_merge(*[pb.io.load(j) for j in jsons])
    else:
        raise ValueError(json)

    class RepresentationPrinter(IPython.lib.pretty.RepresentationPrinter):

        def _enumerate(self, seq):
            """like enumerate, but with an upper limit on the number of items"""
            # Code copied from super()._enumerate
            # Differences:
            #  - Prints a hint for the skipped entries
            #  - Adds a break, when a sequence is longer than
            #    max_seq_length.
            length = 0
            overlength = 0
            for idx, x in enumerate(seq):
                length += 1
                if self.max_seq_length and idx >= self.max_seq_length:
                    overlength += 1
                    # self.text(',')
                    # self.breakable()
                    # self.text('...')
                    # return
                else:
                    yield idx, x
            if overlength:
                self.text(',')
                self.breakable()
                # self.text(f'...,  # {length} - {length-overlength}')
                if color:
                    self.text(
                        f'...,  {_c.Gray}# skipped {overlength} of {length}{_c.Color_Off}')
                else:
                    self.text(
                        f'...,  # skipped {overlength} of {length}')
                p.break_()

    p = RepresentationPrinter(
        sys.stdout, max_width=max_width, max_seq_length=s)

    indent = 4

    @contextlib.contextmanager
    def group_break(p, indent=0, open='', close=''):
        with p.group(indent, open):
            yield
        p.break_()
        if close:
            p.text(close)

    with group_break(p, 4, '{', '}'):
        for k, v in data.items():
            p.break_()
            if k == 'alias':
                k = f'{k!r}: '
                with p.group(len(k), k):
                    p.pretty(v)
                    p.text(',')
            elif k != 'datasets':
                p.text(f'{k!r}: ...,')
            else:
                with group_break(p, indent, f'{k!r}: {{', '},'):
                    for i2, (dataset_name, dataset) in enumerate(v.items()):
                        p.break_()
                        if i2 >= d:
                            p.text(f'{dataset_name!r}: ...,  # {len(dataset)} examples')
                            continue
                        with group_break(p, indent, f'{dataset_name!r}: {{', '},'):
                            for i3, (example_id, example) in enumerate(dataset.items()):
                                if i3 >= n:
                                    break
                                p.break_()
                                with group_break(p, indent, f'{example_id!r}: {{', '},'):
                                    for key, value in example.items():
                                        p.break_()
                                        key = f'{key!r}: '
                                        with p.group(len(key), key):
                                            p.pretty(value)
                                            p.text(f',')

                            skipped = len(dataset) - n
                            if skipped > 0:
                                p.break_()
                                if color:
                                    p.text(f'...,  {_c.Gray}# skipped {skipped} of {len(dataset)} examples{_c.Color_Off}')
                                else:
                                    p.text(f'...,  # skipped {skipped} of {len(dataset)} examples')

    p.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()


def diff(json1, json2, n=1, d=3, max_width=None):
    """
    Use icdiff to print the diff of the previews.

    Args:
        json1:
        json2:
        n:
        d:
        max_width:

    Returns:

    """
    import subprocess
    import shlex
    import tempfile
    from pathlib import Path

    if max_width is None:
        max_width = shutil.get_terminal_size((79, 20)).columns // 2

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        json1 = Path(json1)
        json2 = Path(json2)
        # Add prefix (0_, 1_) to ensure different file names.
        f1 = tmp_dir / ('0_' + json1.name)
        f2 = tmp_dir / ('1_' + json2.name)

        with open(f1, 'w') as fd:
            with contextlib.redirect_stdout(fd):
                preview(json1, d=d, n=n, max_width=max_width)
        with open(f2, 'w') as fd:
            with contextlib.redirect_stdout(fd):
                preview(json2, d=d, n=n, max_width=max_width)

        subprocess.run(
            f'icdiff {shlex.quote(str(f1))} {shlex.quote(str(f2))}',
            shell=True)


def check_audio_exists(json, n=1):
    from pathlib import Path
    from lazy_dataset.database import JsonDatabase
    db = JsonDatabase(json)

    def get_files(o):
        if isinstance(o, (dict, list, tuple)):
            if isinstance(o, dict):
                o = o.items()
            elif isinstance(o, (list, tuple)):
                o = enumerate(o)
            else:
                raise Exception('Cannot happen')
            o = ((f'{k}', v) for k, v in o)
            for k, v in o:
                for sub_k, f in get_files(v):
                    yield (k,) + sub_k, f
        else:
            assert isinstance(o, str), (type(o), o)
            yield (), o

    issue = 0
    for dataset_name in db.dataset_names:
        ds = db.get_dataset(dataset_name)
        print(f'{dataset_name}:')
        for ex in ds[:n]:
            print(f'  {ex["example_id"]}:')
            for k, v in get_files(ex['audio_path']):
                k = '.'.join(k)
                exists = Path(v).exists()
                if not exists:
                    issue += 1
                exists = '✘✔'[exists]
                print(f'    {exists} {k}: {v}')

                # assert Path(v).exists(), (k, v, dataset_name)
            # break
    if issue:
        print(f'ERROR: Found {issue} files that do not exist !!!!!!!!!!!')


if __name__ == '__main__':
    import fire
    fire.Fire({
        'preview': preview,
        'diff': diff,
        'check_audio_exists': check_audio_exists,
    })
