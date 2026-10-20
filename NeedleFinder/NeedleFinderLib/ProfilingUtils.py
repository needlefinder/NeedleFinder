# -*- coding: utf-8 -*-
"""Profiling, logging, and Slicer node lookup helpers for NeedleFinder."""

import inspect
import qt
import slicer

from .LegacyEditorCompat import get_node_or_none

# Global profiling and algorithm configuration flags
profiling = True
loggingInfos = False
frequent = False
MAXNEEDLES = MAXCOL = 206
conesColor = None
outlierThresh_mm = 3


def getNode(pattern=None, index=0, scene=None):
    """Return a MRML node matching pattern, or None if not found (Slicer 4/5 safe)."""
    return get_node_or_none(pattern=pattern, index=index, scene=scene)


def cmp(a, b):
    """Python 3 replacement for built-in cmp()."""
    return (a > b) - (a < b)


def whoami():
    return inspect.stack()[1][3]


def whosdaddy():
    return inspect.stack()[2][3]


def whosgranny():
    return inspect.stack()[3][3]


def lineno():
    """Returns the current line number in our program."""
    return int(inspect.currentframe().f_back.f_lineno)


def pause(box=False):
    if box:
        msgbox("Pause, go to console...")
    try:
        input("You can now inspect the viewers.\nPress enter to continue...")
    except EOFError:
        pass


def msgbox(text):
    print(text)
    qt.QMessageBox.about(None, "Profiling:", text)


def assertbox(cond, msg):
    try:
        assert cond
    except AssertionError:
        breakbox(msg)


def breakbox(text):
    print(text)
    dialog = qt.QDialog()
    ret = qt.QMessageBox.question(
        dialog,
        "Profiling:",
        text + " Continue?",
        qt.QMessageBox.Ok,
        qt.QMessageBox.Cancel,
    )
    if ret != qt.QMessageBox.Ok:
        raise RuntimeError(text)


def profprint(className=""):
    if profiling:
        profString = "%s.%s -----------------------" % (className, whosdaddy())
        print(profString)
        if loggingInfos:
            try:
                with open("/tmp/debug.log", "a") as myfile:
                    myfile.write(profString + "\n")
            except OSError:
                pass


def logprint(className="", msg=""):
    if loggingInfos:
        msg += " from %s.%s" % (className, whosdaddy())
        try:
            with open("/tmp/debug.log", "a") as myfile:
                myfile.write(msg + "\n")
        except OSError:
            pass


def profbox(className=""):
    if profiling:
        strg = "%s.%s -----------------------" % (className, whosdaddy())
        print(strg)
        msgbox(strg)


def getClassName(self):
    return self.__class__.__name__
