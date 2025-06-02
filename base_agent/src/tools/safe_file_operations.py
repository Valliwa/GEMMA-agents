# Create safe_file_operations.py
import os
import logging

class ReadOnlyFileSystem:
    def __init__(self, log_file="readonly_operations.log"):
        self.logger = logging.getLogger("ReadOnlyFS")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def read_file(self, filepath):
        """Actually read files - this is allowed"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            self.logger.info(f"READ: {filepath}")
            return content
        except Exception as e:
            self.logger.error(f"READ FAILED: {filepath} - {e}")
            return None
    
    def write_file(self, filepath, content):
        """Log write attempt but don't actually write"""
        self.logger.info(f"WRITE BLOCKED: {filepath} ({len(content)} characters)")
        return f"[READ-ONLY MODE] Would write {len(content)} characters to {filepath}"
    
    def execute_command(self, command):
        """Log command but don't execute"""
        self.logger.info(f"COMMAND BLOCKED: {command}")
        return f"[READ-ONLY MODE] Would execute: {command}"
    
    def list_directory(self, path):
        """Actually list directories - this is safe"""
        try:
            files = os.listdir(path)
            self.logger.info(f"LIST: {path}")
            return files
        except Exception as e:
            self.logger.error(f"LIST FAILED: {path} - {e}")
            return []
