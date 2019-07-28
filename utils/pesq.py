import os


def pesq(reference, degraded, sample_rate=None, program='pesq', is_current_file=True):
    """ Return PESQ quality estimation (two values: PESQ MOS and MOS LQO) based
    on reference and degraded speech samples comparison.
    Sample rate must be 8000 or 16000 (or can be defined reading reference file
    header).
    PESQ utility must be installed.
    """
    if not os.path.isfile(reference) or not os.path.isfile(degraded):
        raise ValueError('reference or degraded file does not exist')
    if not sample_rate:
        import wave
        w = wave.open(reference, 'r')
        sample_rate = w.getframerate()
        w.close()
    if sample_rate not in (8000, 16000):
        raise ValueError('sample rate must be 8000 or 16000')
    import subprocess
    if is_current_file:
        args = [program, '+%d' % sample_rate, os.getcwd() + '/' + reference, os.getcwd() + '/' + degraded]
    else:
        args = [program, '+%d' % sample_rate, reference, degraded]
    pipe = subprocess.Popen(args, stdout=subprocess.PIPE)
    out, _ = pipe.communicate()
    last_line = out.decode().split('\n')[-2]
    if not last_line.startswith('P.862 Prediction'):
        raise ValueError(last_line)
    return tuple(map(float, last_line.split()[-2:]))
