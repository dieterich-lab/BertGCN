"""Tests for the CLI functionality."""

import sys
from unittest.mock import patch

import pytest

from bertgcn.cli import main


class TestCLIDispatcher:
    """Test the CLI command dispatcher."""

    def test_main_no_args_shows_help(self, capsys):
        """Test that main with no args shows help."""
        # Save original argv
        original_argv = sys.argv

        try:
            # Set argv to simulate no arguments
            sys.argv = ["bertgcn"]

            main()

            captured = capsys.readouterr()
            assert "Usage: bertgcn <command> [args...]" in captured.out
            assert "preprocess" in captured.out
            assert "build-graph" in captured.out
            assert "train" in captured.out
            assert "finetune" in captured.out
            assert "predict" in captured.out
            assert "interpret" in captured.out

        finally:
            # Restore original argv
            sys.argv = original_argv

    @patch("bertgcn.cli.preprocess")
    def test_preprocess_command(self, mock_preprocess):
        """Test preprocess command dispatch."""
        original_argv = sys.argv

        try:
            sys.argv = ["bertgcn", "preprocess"]
            main()

            mock_preprocess.assert_called_once()

        finally:
            sys.argv = original_argv

    @patch("bertgcn.cli.build_graph.main")
    def test_build_graph_command(self, mock_build_graph):
        """Test build-graph command dispatch."""
        original_argv = sys.argv

        try:
            sys.argv = ["bertgcn", "build-graph", "--some", "args"]
            main()

            mock_build_graph.assert_called_once_with(["--some", "args"])

        finally:
            sys.argv = original_argv

    @patch("bertgcn.cli.train_gcn.main")
    def test_train_command(self, mock_train):
        """Test train command dispatch."""
        original_argv = sys.argv

        try:
            sys.argv = ["bertgcn", "train"]
            main()

            mock_train.assert_called_once()

        finally:
            sys.argv = original_argv

    @patch("bertgcn.cli.train_bert.main")
    def test_finetune_command(self, mock_finetune):
        """Test finetune command dispatch."""
        original_argv = sys.argv

        try:
            sys.argv = ["bertgcn", "finetune"]
            main()

            mock_finetune.assert_called_once()

        finally:
            sys.argv = original_argv

    @patch("bertgcn.cli.predict.main")
    def test_predict_command(self, mock_predict):
        """Test predict command dispatch."""
        original_argv = sys.argv

        try:
            sys.argv = ["bertgcn", "predict"]
            main()

            mock_predict.assert_called_once()

        finally:
            sys.argv = original_argv

    @patch("bertgcn.cli.interpret.main")
    def test_interpret_command(self, mock_interpret):
        """Test interpret command dispatch."""
        original_argv = sys.argv

        try:
            sys.argv = ["bertgcn", "interpret"]
            main()

            mock_interpret.assert_called_once()

        finally:
            sys.argv = original_argv

    def test_unknown_command(self, capsys):
        """Test unknown command handling."""
        original_argv = sys.argv

        try:
            sys.argv = ["bertgcn", "unknown_command"]
            main()

            captured = capsys.readouterr()
            # Should show help when command is unknown
            assert "Usage: bertgcn <command> [args...]" in captured.out

        finally:
            sys.argv = original_argv


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def test_all_commands_listed_in_help(self, capsys):
        """Test that all expected commands are listed in help."""
        original_argv = sys.argv

        try:
            sys.argv = ["bertgcn"]
            main()

            captured = capsys.readouterr()
            output = captured.out

            # Check that all commands are mentioned
            expected_commands = [
                "preprocess",
                "build-graph",
                "train",
                "finetune",
                "predict",
                "interpret",
            ]

            for cmd in expected_commands:
                assert cmd in output

        finally:
            sys.argv = original_argv
