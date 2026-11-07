# Brokers one container command's jobserver fifo, argv[1], against the host
# over this process's stdin and stdout, one byte per message:
#   to the host    R  the fifo exists; the command may start
#                  T  the fifo is empty: send a token
#                  +  a token the command returned
#   from the host  +  a token to put in the fifo
#                  F  the command's exec has ended: stop feeding, return every
#                     token written back until the last process holding the
#                     fifo has exited, then remove it and exit
# The host keeps all accounting; this side only moves bytes, keeping at most
# one spare token in the fifo as the host broker does for a local command.
# Linux only: a read from the fifo is seen through inotify, the one readiness
# source that reports it.
import ctypes
import fcntl
import os
import select
import struct
import sys
import termios

IN_ACCESS = 0x1
IN_MODIFY = 0x2

fifo = sys.argv[1]
os.mkfifo(fifo, 0o600)
read_end = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
write_end = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)

libc = ctypes.CDLL(None, use_errno=True)
watcher = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
if watcher < 0:
    raise OSError(ctypes.get_errno(), "inotify_init1 failed")
if libc.inotify_add_watch(watcher, os.fsencode(fifo), IN_ACCESS | IN_MODIFY) < 0:
    raise OSError(ctypes.get_errno(), "inotify_add_watch failed on " + fifo)


def send(message):
    os.write(1, message)


def pending():
    return struct.unpack("i", fcntl.ioctl(read_end, termios.FIONREAD, b"\0\0\0\0"))[0]


def finish():
    os.close(write_end)
    os.set_blocking(read_end, True)
    while True:
        returned = os.read(read_end, 4096)
        if not returned:
            break
        send(b"+" * len(returned))
    os.unlink(fifo)
    sys.exit(0)


requested = False


def reconcile():
    global requested
    spare = 0
    for _ in range(pending() - 1):
        try:
            os.read(read_end, 1)
        except BlockingIOError:
            break
        spare += 1
    if spare:
        send(b"+" * spare)
    if pending() == 0 and not requested:
        send(b"T")
        requested = True


send(b"R")
reconcile()
poller = select.poll()
poller.register(0, select.POLLIN)
poller.register(watcher, select.POLLIN)
while True:
    for fd, _ in poller.poll():
        if fd == watcher:
            try:
                while os.read(watcher, 4096):
                    pass
            except BlockingIOError:
                pass
            continue
        received = os.read(0, 4096)
        if not received:
            # The host is gone and has settled the command's tokens itself.
            os.unlink(fifo)
            sys.exit(0)
        for message in received:
            if message == ord("+"):
                os.write(write_end, b"+")
                requested = False
            elif message == ord("F"):
                finish()
            else:
                sys.exit("unknown message from the host: %r" % bytes([message]))
    reconcile()
