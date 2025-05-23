# src/utils/error_handling.py
import logging
import traceback 
import time 
from functools import wraps 
from typing import Any, Callable, Optional, Tuple
import os
# Imports for SystemHealthChecker
import psutil # Needs to be available or placeholder
# import torch # Placeholder, import dynamically in method
# import pyaudio # Placeholder, import dynamically in method


# Custom Exception Classes (ensure these are as defined previously)
class NHSNavigatorError(Exception): pass # Base exception
class ModelLoadError(NHSNavigatorError): pass # Error loading AI model
class VoiceSystemError(NHSNavigatorError): pass # Voice system errors
class AvatarError(NHSNavigatorError): pass # Avatar system errors
class ConfigurationError(NHSNavigatorError): pass # Configuration problems

def setup_logging(log_file='logs/nhs_navigator.log', audit_log_file='logs/nhs_navigator_audit.log', level=logging.INFO) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(os.path.dirname(audit_log_file), exist_ok=True)

    # Main application logger
    logger = logging.getLogger('NHS_Navigator')
    if not logger.handlers: 
        logger.setLevel(level)
        # File Handler for main log
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        # Console Handler for main log
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
        fh.setFormatter(formatter); ch.setFormatter(formatter)
        logger.addHandler(fh); logger.addHandler(ch)
        # logger.info("Main application logging setup complete.") # Can be noisy

    # Audit Logger
    audit_logger = logging.getLogger('NHS_Navigator_Audit')
    if not audit_logger.handlers:
        audit_logger.setLevel(logging.INFO) # Audit all INFO level messages
        # File Handler for audit log
        afh = logging.FileHandler(audit_log_file)
        afh.setLevel(logging.INFO)
        audit_formatter = logging.Formatter('%(asctime)s - AUDIT - %(message)s') # Simpler format for audit
        afh.setFormatter(audit_formatter)
        audit_logger.addHandler(afh)
        # audit_logger.info("Audit logging setup complete.") # Can be noisy

    return logger # Return main logger by default

class ErrorHandler:
    def __init__(self):
        self.logger = setup_logging() 
        self.audit_logger = logging.getLogger('NHS_Navigator_Audit') # Get audit logger instance
        self.logger.info("ErrorHandler initialized (Phase 9). Audit logger also available.")
        self.fallback_guidance = {
            "general_nhs_info": "For general NHS information, please visit the official NHS website.",
            "ae_questions": "In A&E, staff typically ask about your symptoms, when they started, medical history, allergies, and current medications.",
            "technical_difficulty": "We are experiencing technical difficulties. Please try again later."
        }
    def handle_model_error(self, error: Exception, context: str = "") -> str:
        self.logger.error(f"MODEL_ERROR ({context}): {str(error)}", exc_info=True)
        return self.fallback_guidance["technical_difficulty"] + " (AI Model Error)"

    def handle_voice_error(self, error: Exception, context: str = "") -> str:
        self.logger.error(f"VOICE_SYSTEM_ERROR ({context}): {str(error)}", exc_info=True)
        return "Voice system error. Please type your query."

    def handle_avatar_error(self, error: Exception, context: str = "") -> str:
        self.logger.error(f"AVATAR_ERROR ({context}): {str(error)}", exc_info=True)
        return "Avatar display error."

    def get_user_friendly_message(self, error: Exception, context: str = "") -> str:
        self.logger.warning(f"UserFriendlyMsg for ({context}): {type(error).__name__} - {str(error)}")
        if isinstance(error, ModelLoadError): return self.handle_model_error(error, f"UF-{context}")
        if isinstance(error, VoiceSystemError): return self.handle_voice_error(error, f"UF-{context}")
        if isinstance(error, AvatarError): return self.handle_avatar_error(error, f"UF-{context}")
        if isinstance(error, ConfigurationError):
             self.logger.error(f"CONFIGURATION_ERROR ({context}): {str(error)}", exc_info=True)
             return "There is a configuration problem with the application. Please contact support."
        err_str = str(error).lower()
        if "network" in err_str or "connection" in err_str:
            return "A network connection issue occurred. Please check your internet and try again."
        return self.fallback_guidance["technical_difficulty"]


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, errors_to_catch: Tuple[type[Exception], ...] = (Exception,)):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = setup_logging() 
            for attempt in range(max_retries):
                try: return func(*args, **kwargs)
                except errors_to_catch as e:
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__} failed: {type(e).__name__} - {e}")
                    if attempt == max_retries - 1: 
                        logger.error(f"All {max_retries} retries failed for function {func.__name__}.", exc_info=True)
                        raise 
                    time.sleep(delay * (2 ** attempt)) 
            return None 
        return wrapper
    return decorator

def safe_execute(func: Callable, fallback_value: Any = None, log_errors: bool = True, error_handler_instance: Optional[ErrorHandler] = None, context: str = "") -> Any:
    logger = setup_logging() 
    try: return func()
    except Exception as e:
        if log_errors:
            logger.error(f"Safe execution of {func.__name__} (context: {context}) failed: {type(e).__name__} - {e}", exc_info=True)
            if error_handler_instance:
                user_message = error_handler_instance.get_user_friendly_message(e, f"SafeExecute-{func.__name__}-{context}")
                logger.warning(f"User-friendly message for safe_execute failure of {func.__name__}: {user_message}")
        return fallback_value

class SystemHealthChecker:
    def __init__(self):
        self.logger = setup_logging() # Use existing setup_logging
        self.logger.info("SystemHealthChecker initialized.")

    def check_system_health(self) -> dict:
        self.logger.info("Performing system health check...")
        health_status = {
            "overall": "healthy", # healthy, warning, critical
            "components": {},
            "warnings": [],
            "errors": []
        }

        # 1. GPU Check (Placeholder)
        try:
            import torch # Try import inside method
            if torch.cuda.is_available():
                health_status["components"]["gpu"] = {"status": "available", "devices": torch.cuda.device_count()}
                # Basic memory check
                # total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                # allocated_mem_gb = torch.cuda.memory_allocated(0) / (1024**3)
                # if allocated_mem_gb / total_mem_gb > 0.9: # More than 90% allocated
                #    health_status["warnings"].append(f"GPU memory usage high: {allocated_mem_gb:.2f}/{total_mem_gb:.2f} GB")
            else:
                health_status["components"]["gpu"] = {"status": "unavailable"}
                health_status["warnings"].append("GPU (CUDA) not available. AI performance may be impacted.")
        except ImportError:
            health_status["components"]["gpu"] = {"status": "pytorch_not_installed"}
            health_status["warnings"].append("PyTorch not found, cannot check GPU status.")
        except Exception as e:
            health_status["components"]["gpu"] = {"status": "error", "details": str(e)}
            health_status["errors"].append(f"GPU check failed: {str(e)}")

        # 2. Audio System Check (Placeholder)
        try:
            import pyaudio # Try import inside method
            p = pyaudio.PyAudio()
            device_count = p.get_device_count()
            health_status["components"]["audio_system"] = {"status": "available", "devices": device_count}
            if device_count == 0:
                health_status["warnings"].append("No audio input/output devices found by PyAudio.")
            p.terminate()
        except ImportError:
            health_status["components"]["audio_system"] = {"status": "pyaudio_not_installed"}
            health_status["warnings"].append("PyAudio not found, cannot check audio system.")
        except Exception as e:
            health_status["components"]["audio_system"] = {"status": "error", "details": str(e)}
            health_status["errors"].append(f"Audio system check failed: {str(e)}")

        # 3. Memory Usage Check
        try:
            # import psutil # Already imported at top level of this file
            mem = psutil.virtual_memory()
            health_status["components"]["memory"] = {"total_gb": f"{mem.total / (1024**3):.2f}", "available_gb": f"{mem.available / (1024**3):.2f}", "percent_used": mem.percent}
            if mem.percent > 90:
                health_status["warnings"].append(f"High system memory usage: {mem.percent}%")
        # except ImportError: # psutil is imported at the top
        #     health_status["components"]["memory"] = {"status": "psutil_not_installed"}
        #     health_status["warnings"].append("psutil not found, cannot check system memory.")
        except Exception as e:
            health_status["errors"].append(f"Memory check failed: {str(e)}")
        
        # 4. Critical File/Model Path Check (Placeholder)
        model_path = "./models/lora_adapters" # Example path
        if os.path.exists(model_path) and os.path.isdir(model_path):
            health_status["components"]["model_files"] = {"status": "present", "path": model_path}
        else:
            health_status["components"]["model_files"] = {"status": "missing", "path": model_path}
            health_status["warnings"].append(f"Critical model directory missing: {model_path}")

        # Determine overall status
        if health_status["errors"]:
            health_status["overall"] = "critical"
        elif health_status["warnings"]:
            health_status["overall"] = "warning"
        
        self.logger.info(f"System health check complete. Overall: {health_status['overall']}")
        return health_status

    def create_diagnostic_report(self) -> str:
        health = self.check_system_health()
        report = ["--- NHS Navigator Diagnostic Report ---"]
        report.append(f"Timestamp: {time.ctime()}")
        report.append(f"Overall Status: {health['overall'].upper()}")
        report.append("\nComponents:")
        for comp, data in health['components'].items():
            report.append(f"  - {comp.replace('_', ' ').title()}: {data}")
        if health['warnings']:
            report.append("\nWarnings:")
            for warn in health['warnings']: report.append(f"  [!] {warn}")
        if health['errors']:
            report.append("\nErrors:")
            for err in health['errors']: report.append(f"  [X] {err}")
        return "\n".join(report)
    
if __name__ == "__main__":
    main_logger = setup_logging() 
    audit_logger_instance = logging.getLogger('NHS_Navigator_Audit')
    main_logger.info("error_handling.py self-test (Phase 10 - includes SystemHealthChecker).")
    audit_logger_instance.info("SELF_TEST_EVENT - Type: SystemStart, User: system, Details: error_handling.py main block executed (Phase 10).")
    
    eh = ErrorHandler()
    try:
        raise ModelLoadError("Simulated model load failure for audit test.")
    except ModelLoadError as e_mle:
        user_msg = eh.get_user_friendly_message(e_mle, "ErrorHandlerSelfTest")
        main_logger.info(f"Self-test handled error: {user_msg}")
    
    main_logger.info("SystemHealthChecker self-test started.")
    shc = SystemHealthChecker()
    print(shc.create_diagnostic_report())
    main_logger.info("SystemHealthChecker self-test finished.")
    
    main_logger.info("error_handling.py self-test (Phase 10) finished.")
