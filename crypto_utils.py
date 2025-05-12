
# /crypto_utils.py
import os
import base64
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import re

class CryptoUtils:
    def __init__(self, key_file="crypto.key", password=None):
        """Initialize crypto utilities with a key file or password"""
        self.key_file = key_file
        self.password = password
        self.cipher_suite = self._get_cipher_suite()
    
    def _get_cipher_suite(self):
        """Get or generate a Fernet cipher suite"""
        key = self._get_key()
        return Fernet(key)
    
    def _get_key(self):
        """Get key from file or generate from password"""
        if self.password:
            # Derive key from password
            salt = b'8x4n8lz9q7p1o3m5' # Fixed salt, you can make this configurable
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.password.encode()))
            return key
        else:
            # Get key from file or generate new one
            try:
                with open(self.key_file, "rb") as key_file:
                    key = key_file.read()
                    return key
            except FileNotFoundError:
                key = Fernet.generate_key()
                with open(self.key_file, "wb") as key_file:
                    key_file.write(key)
                return key
    
    def encrypt_text(self, text):
        """Encrypt a string"""
        if not isinstance(text, str):
            return ""
        
        # Check if the text needs to be encrypted (looking for potentially sensitive content)
        if not self._should_encrypt(text):
            return text
        
        encrypted = self.cipher_suite.encrypt(text.encode())
        # Convert to URL-safe base64 string for storage
        return f"ENC:{base64.urlsafe_b64encode(encrypted).decode()}"
    
    def decrypt_text(self, encrypted_text):
        """Decrypt a string that was encrypted with encrypt_text"""
        if not isinstance(encrypted_text, str):
            return ""
        
        # Check if it's actually encrypted
        if not encrypted_text.startswith("ENC:"):
            return encrypted_text
        
        try:
            # Remove the ENC: prefix and decode
            encoded_text = encrypted_text[4:]
            decoded = base64.urlsafe_b64decode(encoded_text)
            decrypted = self.cipher_suite.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            print(f"Decryption error: {e}")
            return f"[DECRYPTION ERROR: {encrypted_text[:20]}...]"
    
    def _should_encrypt(self, text):
        """
        Determine if text contains sensitive content that should be encrypted
        Using simple pattern matching for demonstration
        """
        # List of patterns for potentially sensitive content
        patterns = [
            # Credit card numbers
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            # Social security numbers
            r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
            # Explicit content keywords - simplified for demonstration
            r'\b(?:password|secret|credential|private|nsfw|hack|exploit)\b',
            # IP addresses
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            # Email addresses
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def encrypt_file(self, input_file, output_file=None):
        """Encrypt an entire file"""
        if output_file is None:
            output_file = input_file + ".enc"
        
        try:
            with open(input_file, 'rb') as f:
                data = f.read()
            
            encrypted_data = self.cipher_suite.encrypt(data)
            
            with open(output_file, 'wb') as f:
                f.write(encrypted_data)
            
            return True, output_file
        except Exception as e:
            return False, str(e)
    
    def decrypt_file(self, input_file, output_file=None):
        """Decrypt an encrypted file"""
        if output_file is None:
            if input_file.endswith('.enc'):
                output_file = input_file[:-4]
            else:
                output_file = input_file + ".dec"
        
        try:
            with open(input_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            
            with open(output_file, 'wb') as f:
                f.write(decrypted_data)
            
            return True, output_file
        except Exception as e:
            return False, str(e)

# Simple functions for direct use
def encrypt_text(text, key_file="crypto.key"):
    """Simple wrapper for encrypting text"""
    crypto = CryptoUtils(key_file)
    return crypto.encrypt_text(text)

def decrypt_text(encrypted_text, key_file="crypto.key"):
    """Simple wrapper for decrypting text"""
    crypto = CryptoUtils(key_file)
    return crypto.decrypt_text(encrypted_text)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Encrypt or decrypt files and text")
    parser.add_argument("--mode", choices=["encrypt", "decrypt"], required=True, help="Operation mode")
    parser.add_argument("--type", choices=["file", "text"], required=True, help="Input type")
    parser.add_argument("--input", required=True, help="Input file or text")
    parser.add_argument("--output", help="Output file (for file mode)")
    parser.add_argument("--key", help="Key file path (optional)")
    parser.add_argument("--password", help="Password for encryption/decryption (instead of key file)")
    
    args = parser.parse_args()
    
    # Create crypto utils instance
    crypto = CryptoUtils(key_file=args.key, password=args.password)
    
    if args.type == "file":
        if args.mode == "encrypt":
            success, result = crypto.encrypt_file(args.input, args.output)
        else:  # decrypt
            success, result = crypto.decrypt_file(args.input, args.output)
        
        if success:
            print(f"Successfully {'encrypted' if args.mode == 'encrypt' else 'decrypted'} to: {result}")
        else:
            print(f"Failed: {result}")
    
    else:  # text mode
        if args.mode == "encrypt":
            result = crypto.encrypt_text(args.input)
            print(f"Encrypted: {result}")
        else:  # decrypt
            result = crypto.decrypt_text(args.input)
            print(f"Decrypted: {result}")
