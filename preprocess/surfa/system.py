import os
import sys
import platform
import subprocess as sp
import warnings


# reformat default warning message to make it a bit more mininalistic
def formatwarning(message, category, filename, lineno, line=None):
    return f'{category.__name__}: {message}\n'
warnings.formatwarning = formatwarning


def run(command, silent=False, background=False, executable='/bin/bash', log=None):
    """
    Runs a shell command and returns the exit code.

    Parameters
    ----------
    command : str
        Command to run.
    silent : bool
        Send output to devnull.
    background : bool
        Run command as a background process.
    executable : str
        Shell executable. Defaults to bash.
    log : str
        Send output to a log file.

    Returns
    -------
    int
        Command exit code.
    """

    # redirect the standard output appropriately
    if silent:
        std = {'stdout': sp.DEVNULL, 'stderr': sp.DEVNULL}
    elif not background:
        std = {'stdout': sp.PIPE, 'stderr': sp.STDOUT}
    else:
        std = {}  # do not redirect

    # run the command
    process = sp.Popen(command, **std, shell=True, executable=executable)
    if not background:
        # write the standard output stream
        if process.stdout:
            for line in process.stdout:
                decoded = line.decode('utf-8')
                if log is not None:
                    with open(log, 'a') as file:
                        file.write(decoded)
                sys.stdout.write(decoded)
        # wait for process to finish
        process.wait()

    return process.returncode


def collect_output(command, executable='/bin/bash'):
    """
    Collect the output of a shell command.

    Parameters
    ----------
    command : str
        Command to run.
    executable : str
        Shell executable. Defaults to bash.

    Returns
    -------
    tuple of (str, int)
        Tuple containing the command output and the corresponding exit code.
    """
    result = sp.run(command, stdout=sp.PIPE, stderr=sp.STDOUT, shell=True, executable=executable)
    return (result.stdout.decode('utf-8'), result.returncode)


def hostname(short=True):
    """
    Get the system hostname.

    Parameters
    ----------
        short: Provide the short hostname. Defaults to True.
    """
    node = platform.node()
    if short:
        return node.split('.')[0]
    return node


def vmpeak():
    """
    Return the peak memory usage of the process in kilobytes.

    Note: This only works on linux machines because it requires `/proc/self/status`.
    """
    # TODO: switch to this (portable across platforms)
    # return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    procstatus = '/proc/self/status'
    if os.path.exists(procstatus):
        with open(procstatus, 'r') as file:
            for line in file:
                if 'VmPeak' in line:
                    return int(line.split()[1])
    return None


def fatal(message, retcode=1):
    """
    Print an error message and exit (or raise an exception if in interactive mode).

    Parameters
    ----------
    message : str
        Error message to print
    retcode : int
        Exit code. Defaults to 1.
    """
    import __main__ as main
    if hasattr(main, '__file__'):
        print(f'Error: {message}')
        sys.exit(retcode)
    else:
        raise Exception(message)


def readlines(filename):
    """
    Read the lines of a text file.

    Parameters
    ----------
    filename : str
        Text file to read.

    Returns
    -------
    content : list
        List of stripped lines in text file.
    """
    with open(filename) as file:
        content = file.read().splitlines()
    return content
