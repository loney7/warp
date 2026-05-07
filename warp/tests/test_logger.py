# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import sys
import unittest
import warnings

import warp
import warp as wp
from warp._src import logger as _logger_module
from warp._src.logger import LoggerBasic, log_debug, log_error, log_warning


class TestLogger(unittest.TestCase):
    def test_log_level_constants(self):
        self.assertEqual(wp.LOG_DEBUG, 10)
        self.assertEqual(wp.LOG_INFO, 20)
        self.assertEqual(wp.LOG_WARNING, 30)
        self.assertEqual(wp.LOG_ERROR, 40)

    def test_logger_protocol_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            wp.Logger()

    def test_set_logger_accepts_duck_typed_object(self):
        """Logger is a Protocol -- any object with the four methods works."""

        class DuckLogger:
            def debug(self, message):
                pass

            def info(self, message):
                pass

            def warning(self, message, category=None, stacklevel=1):
                pass

            def error(self, message):
                pass

        original = wp.get_logger()
        try:
            wp.set_logger(DuckLogger())
            self.assertIsInstance(wp.get_logger(), DuckLogger)
        finally:
            wp.set_logger(original)

    def test_set_logger_rejects_object_missing_methods(self):
        class NotALogger:
            def debug(self, message):
                pass

            # missing info, warning, error

        with self.assertRaises(TypeError):
            wp.set_logger(NotALogger())

    def test_basic_logger_debug_writes_to_stdout(self):
        logger = LoggerBasic()
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            logger.debug("test debug msg")
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        self.assertEqual(output, "test debug msg\n")

    def test_basic_logger_info_writes_to_stdout(self):
        logger = LoggerBasic()
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            logger.info("test info msg")
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        self.assertEqual(output, "test info msg\n")

    def test_basic_logger_error_writes_to_stderr(self):
        logger = LoggerBasic()
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            logger.error("something broke")
            output = sys.stderr.getvalue()
        finally:
            sys.stderr = old_stderr
        self.assertEqual(output, "Warp Error: something broke\n")

    def test_basic_logger_warning_respects_filters(self):
        """GH-1315: user warning filters must not be overridden."""
        logger = LoggerBasic()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                logger.warning("old API", category=DeprecationWarning, stacklevel=1)
                output = sys.stderr.getvalue()
            finally:
                sys.stderr = old_stderr
        self.assertEqual(output, "", "DeprecationWarning should have been suppressed")

    def test_set_get_logger(self):
        original = wp.get_logger()
        try:
            self.assertIsInstance(original, LoggerBasic)
            kit = wp.utils.LoggerKit()
            wp.set_logger(kit)
            self.assertIs(wp.get_logger(), kit)
        finally:
            wp.set_logger(original)

    def test_set_logger_validates_type(self):
        with self.assertRaises(TypeError):
            wp.set_logger("not a logger")

    def test_set_logger_none_resets_to_basic_logger(self):
        original = wp.get_logger()
        try:
            wp.set_logger(wp.utils.LoggerKit())
            self.assertIsInstance(wp.get_logger(), wp.utils.LoggerKit)
            wp.set_logger(None)
            self.assertIsInstance(wp.get_logger(), LoggerBasic)
        finally:
            wp.set_logger(original)

    def test_scoped_logger_swaps_and_restores(self):
        original = wp.get_logger()
        kit = wp.utils.LoggerKit()
        with wp.ScopedLogger(kit) as scope:
            self.assertIs(scope.logger, kit)
            self.assertIs(wp.get_logger(), kit)
        self.assertIs(wp.get_logger(), original)

    def test_scoped_logger_restores_on_exception(self):
        original = wp.get_logger()
        with self.assertRaises(RuntimeError):
            with wp.ScopedLogger(wp.utils.LoggerKit()):
                raise RuntimeError("boom")
        self.assertIs(wp.get_logger(), original)

    def test_scoped_logger_none_uses_basic_logger(self):
        original = wp.get_logger()
        wp.set_logger(wp.utils.LoggerKit())
        try:
            with wp.ScopedLogger(None):
                self.assertIsInstance(wp.get_logger(), LoggerBasic)
        finally:
            wp.set_logger(original)

    def test_scoped_log_level_swaps_and_restores(self):
        original = wp.config.log_level
        try:
            wp.config.log_level = wp.LOG_INFO
            with wp.ScopedLogLevel(wp.LOG_ERROR):
                self.assertEqual(wp.config.log_level, wp.LOG_ERROR)
            self.assertEqual(wp.config.log_level, wp.LOG_INFO)
        finally:
            wp.config.log_level = original

    def test_scoped_log_level_restores_on_exception(self):
        original = wp.config.log_level
        try:
            wp.config.log_level = wp.LOG_INFO
            with self.assertRaises(RuntimeError):
                with wp.ScopedLogLevel(wp.LOG_ERROR):
                    raise RuntimeError("boom")
            self.assertEqual(wp.config.log_level, wp.LOG_INFO)
        finally:
            wp.config.log_level = original

    def test_kit_logger_carb_path_uses_warp_prefix(self):
        """LoggerKit formats messages with the same Warp-branded prefix as LoggerBasic."""
        captured = []

        class FakeCarb:
            def log_verbose(self, msg):
                captured.append(("verbose", msg))

            def log_info(self, msg):
                captured.append(("info", msg))

            def log_warn(self, msg):
                captured.append(("warn", msg))

            def log_error(self, msg):
                captured.append(("error", msg))

        logger = wp.utils.LoggerKit()
        original_carb = logger._carb
        logger._carb = FakeCarb
        try:
            logger.warning("oops", category=DeprecationWarning)
            logger.warning("nocat")
            logger.error("broken")
        finally:
            logger._carb = original_carb
        self.assertEqual(captured[0], ("warn", "Warp DeprecationWarning: oops"))
        self.assertEqual(captured[1], ("warn", "Warp UserWarning: nocat"))
        self.assertEqual(captured[2], ("error", "Warp Error: broken"))

    def test_log_debug_gated_by_level(self):
        original_level = warp.config.log_level
        warp.config.log_level = wp.LOG_WARNING
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            log_debug("should not appear")
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
            warp.config.log_level = original_level
        self.assertEqual(output, "")

    def test_log_debug_emits_at_debug_level(self):
        original_level = warp.config.log_level
        warp.config.log_level = wp.LOG_DEBUG
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            log_debug("debug msg")
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
            warp.config.log_level = original_level
        self.assertEqual(output, "debug msg\n")

    def test_log_error_always_emits(self):
        """log_error ignores log_level -- errors are never suppressed."""
        original_level = warp.config.log_level
        warp.config.log_level = wp.LOG_ERROR + 10
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            log_error("critical failure")
            output = sys.stderr.getvalue()
        finally:
            sys.stderr = old_stderr
            warp.config.log_level = original_level
        self.assertIn("critical failure", output)

    def test_log_warning_once_deduplicates(self):
        with warnings.catch_warnings():
            warnings.resetwarnings()
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                log_warning("dup warning", category=UserWarning, once=True)
                log_warning("dup warning", category=UserWarning, once=True)
                output = sys.stderr.getvalue()
            finally:
                sys.stderr = old_stderr
        self.assertEqual(output.count("dup warning"), 1)

    def test_kit_logger_routes_to_stdout(self):
        logger = wp.utils.LoggerKit()
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            logger.debug("kit debug")
            logger.info("kit info")
            logger.error("kit error")
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        self.assertIn("kit debug", output)
        self.assertIn("kit info", output)
        self.assertIn("Warp Error: kit error", output)

    def test_log_warning_deprecation_warnings_deduplicate_without_once(self):
        """DeprecationWarnings are deduplicated even when ``once`` is not passed,
        matching the legacy ``warn()`` helper this replaced."""
        saved = _logger_module._warnings_seen.copy()
        _logger_module._warnings_seen.clear()
        with warnings.catch_warnings():
            warnings.resetwarnings()
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                log_warning("legacy api", category=DeprecationWarning)
                log_warning("legacy api", category=DeprecationWarning)
                output = sys.stderr.getvalue()
            finally:
                sys.stderr = old_stderr
                _logger_module._warnings_seen.clear()
                _logger_module._warnings_seen.update(saved)
        self.assertEqual(output.count("legacy api"), 1)

    def test_scoped_logger_exit_tolerates_mutated_saved_logger(self):
        """``ScopedLogger.__exit__`` must restore the saved logger without
        re-validating it; otherwise a TypeError would mask any in-flight
        exception propagating through the context."""

        class MutableLogger:
            def debug(self, message):
                pass

            def info(self, message):
                pass

            def warning(self, message, category=None, stacklevel=1):
                pass

            def error(self, message):
                pass

        original = wp.get_logger()
        outer = MutableLogger()
        wp.set_logger(outer)
        try:
            with wp.ScopedLogger(MutableLogger()):
                # Break the saved logger after entering the scope; __exit__
                # must still restore it without raising.
                outer.debug = None
            self.assertIs(wp.get_logger(), outer)
        finally:
            wp.set_logger(original)

    def test_gh1315_user_filters_respected(self):
        """Verify that warnings.filterwarnings('ignore') suppresses Warp warnings."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                log_warning("deprecated thing", category=DeprecationWarning)
                output = sys.stderr.getvalue()
            finally:
                sys.stderr = old_stderr
        self.assertEqual(output, "", "DeprecationWarning should have been suppressed by user filter")


if __name__ == "__main__":
    unittest.main()
