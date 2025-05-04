"""
Tests for the logger module.
"""

import pytest
import logging
import json
from pathlib import Path
from src.logger import setup_logging, get_logger, CustomJsonFormatter

def test_setup_logging(test_data_dir):
    """Test setup_logging function."""
    # Setup logging with test directory
    log_dir = test_data_dir / 'logs'
    setup_logging(log_level='DEBUG', log_dir=str(log_dir))
    
    # Get root logger
    logger = logging.getLogger()
    
    # Check logger configuration
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 2  # Console and file handlers
    
    # Check file handler
    file_handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
    assert file_handler.baseFilename.startswith(str(log_dir))
    assert isinstance(file_handler.formatter, CustomJsonFormatter)
    
    # Check console handler
    console_handler = next(h for h in logger.handlers if isinstance(h, logging.StreamHandler))
    assert isinstance(console_handler.formatter, CustomJsonFormatter)

def test_get_logger():
    """Test get_logger function."""
    # Get logger for a specific module
    logger = get_logger('test_module')
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == 'test_module'
    assert logger.level == logging.INFO  # Default level
    assert len(logger.handlers) == 0  # Should not have handlers

def test_custom_json_formatter():
    """Test CustomJsonFormatter class."""
    # Create formatter
    formatter = CustomJsonFormatter()
    
    # Create a log record
    record = logging.LogRecord(
        name='test',
        level=logging.INFO,
        pathname='test.py',
        lineno=1,
        msg='Test message',
        args=(),
        exc_info=None
    )
    
    # Format the record
    formatted = formatter.format(record)
    parsed = json.loads(formatted)
    
    # Check formatted output
    assert isinstance(parsed, dict)
    assert 'timestamp' in parsed
    assert 'level' in parsed
    assert 'module' in parsed
    assert 'function' in parsed
    assert 'line' in parsed
    assert 'message' in parsed
    assert parsed['message'] == 'Test message'
    assert parsed['level'] == 'INFO'

def test_logging_output(test_data_dir):
    """Test actual logging output."""
    # Setup logging
    log_dir = test_data_dir / 'logs'
    setup_logging(log_level='INFO', log_dir=str(log_dir))
    
    # Get logger
    logger = get_logger('test_module')
    
    # Add file handler
    log_file = log_dir / 'test.log'
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(CustomJsonFormatter())
    logger.addHandler(file_handler)
    
    # Log some messages
    logger.info('Test info message')
    logger.warning('Test warning message')
    logger.error('Test error message')
    
    # Read log file
    with open(log_file, 'r') as f:
        log_lines = f.readlines()
    
    # Check log output
    assert len(log_lines) == 3
    
    # Parse and check each log entry
    for line in log_lines:
        log_entry = json.loads(line)
        assert 'timestamp' in log_entry
        assert 'level' in log_entry
        assert 'message' in log_entry
        assert log_entry['message'].startswith('Test')

def test_log_levels():
    """Test different log levels."""
    # Setup logging with DEBUG level
    setup_logging(log_level='DEBUG')
    
    # Get logger
    logger = get_logger('test_module')
    
    # Test all log levels
    levels = ['debug', 'info', 'warning', 'error', 'critical']
    for level in levels:
        log_func = getattr(logger, level)
        log_func(f'Test {level} message')
    
    # Check that all levels are properly handled
    assert logger.isEnabledFor(logging.DEBUG)
    assert logger.isEnabledFor(logging.INFO)
    assert logger.isEnabledFor(logging.WARNING)
    assert logger.isEnabledFor(logging.ERROR)
    assert logger.isEnabledFor(logging.CRITICAL)

def test_log_rotation(test_data_dir):
    """Test log file rotation."""
    # Setup logging with small maxBytes to force rotation
    log_dir = test_data_dir / 'logs'
    setup_logging(log_level='INFO', log_dir=str(log_dir))
    
    # Get logger
    logger = get_logger('test_module')
    
    # Add rotating file handler
    log_file = log_dir / 'rotation_test.log'
    file_handler = logging.handlers.RotatingFileHandler(
        str(log_file),
        maxBytes=100,  # Small size to force rotation
        backupCount=3
    )
    file_handler.setFormatter(CustomJsonFormatter())
    logger.addHandler(file_handler)
    
    # Log many messages to force rotation
    for i in range(100):
        logger.info(f'Test message {i}')
    
    # Check that rotation files were created
    assert log_file.exists()
    assert (log_dir / 'rotation_test.log.1').exists()
    assert (log_dir / 'rotation_test.log.2').exists()
    assert (log_dir / 'rotation_test.log.3').exists()
    assert not (log_dir / 'rotation_test.log.4').exists()  # Should not exceed backupCount 