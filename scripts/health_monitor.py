# scripts/health_monitor.py
import time
import os
import sys
import logging # Added for audit_logger instance
# Ensure src is in path for imports if this script is run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.utils.error_handling import setup_logging, SystemHealthChecker # Assuming SystemHealthChecker is now in error_handling
    from src.inference.nhs_assistant import NHSAssistant # For AI model responsiveness check
    # Ensure NHSAssistant and its dependencies (ModelLoader, ModelConfig) have dummy versions for standalone script run if needed
except ImportError as e:
    print(f"Failed to import modules for health_monitor.py: {e}")
    # Define dummy classes if imports fail
    # import logging # logging already imported
    def setup_logging(): logger_instance = logging.getLogger("DummyHealthMonitor"); logger_instance.addHandler(logging.NullHandler()); return logger_instance
    class SystemHealthChecker:
        def check_system_health(self): return {"overall": "dummy_error", "errors": ["Dummy SHC error"], "warnings":[]}
        def create_diagnostic_report(self): return "Dummy diagnostic report"
    class NHSAssistant:
        def __init__(self): self.is_initialized = False; self.logger = setup_logging()
        def initialize(self): self.logger.info("Dummy NHSAssistant init for health check."); self.is_initialized = True
        def get_response(self, query): 
            if query == "hello_nhs_health_check": return "Hello! Assistant is responsive."
            return "Dummy fallback response from health check assistant."

logger = setup_logging()
audit_logger = logging.getLogger('NHS_Navigator_Audit') # Get audit logger

def check_ai_model_responsiveness(assistant: NHSAssistant) -> bool:
    logger.info("Checking AI model responsiveness...")
    if not assistant.is_initialized:
        logger.warning("AI Assistant not initialized, attempting to initialize for health check.")
        try:
            assistant.initialize() # Assistant's own init logs to audit
            if not assistant.is_initialized:
                logger.error("AI Assistant failed to initialize during health check.")
                audit_logger.error("HEALTH_CHECK_EVENT - Type: AIModelResponsiveness, Status: Failed, Reason: AssistantInitFail")
                return False
        except Exception as e:
            logger.error(f"Exception during AI Assistant initialization for health check: {e}", exc_info=True)
            audit_logger.error(f"HEALTH_CHECK_EVENT - Type: AIModelResponsiveness, Status: Failed, Reason: AssistantInitException - {type(e).__name__}")
            return False
    
    try:
        start_time = time.time()
        # Use a specific, simple, non-critical query for health check
        response = assistant.get_response("hello_nhs_health_check")
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        known_fallbacks = ["technical difficulties", "AI assistant is currently unavailable", "AI model failed", "Assistant not ready", "dummy fallback"]
        is_fallback = any(fb.lower() in response.lower() for fb in known_fallbacks)

        if not is_fallback and len(response) > 10: # Arbitrary length for a "real" response
            logger.info(f"AI model responsive. Latency: {response_time_ms:.2f} ms.")
            audit_logger.info(f"HEALTH_CHECK_EVENT - Type: AIModelResponsiveness, Status: OK, LatencyMs: {response_time_ms:.2f}")
            return True
        else:
            logger.warning(f"AI model responded with fallback or short message: '{response}'. Latency: {response_time_ms:.2f} ms.")
            audit_logger.warning(f"HEALTH_CHECK_EVENT - Type: AIModelResponsiveness, Status: UnresponsiveOrFallback, LatencyMs: {response_time_ms:.2f}, ResponseSample: '{response[:50]}...'")
            return False
    except Exception as e:
        logger.error(f"Error during AI model responsiveness check: {e}", exc_info=True)
        audit_logger.error(f"HEALTH_CHECK_EVENT - Type: AIModelResponsiveness, Status: Failed, Reason: Exception - {type(e).__name__}")
        return False

def track_performance_metrics(assistant: NHSAssistant): # Pass assistant for specific checks
    logger.info("Tracking performance metrics (simulated for now)...")
    # This is where more detailed metrics could be logged if needed.
    # For example, using psutil for disk space as conceptualized:
    # try:
    #     import psutil
    #     disk_space = psutil.disk_usage('/').percent
    #     logger.info(f"Current disk space usage: {disk_space}%")
    #     audit_logger.info(f"PERF_METRIC - Type: DiskUsagePercent, Value: {disk_space}")
    # except ImportError:
    #     logger.warning("psutil not available, cannot track disk space metric for health monitor.")
    # except Exception as e:
    #     logger.error(f"Failed to track disk space metric: {e}")
    pass


def run_health_monitoring_loop(interval_seconds=300): # Default 5 mins
    logger.info(f"Starting health monitoring loop (interval: {interval_seconds}s).")
    health_checker = SystemHealthChecker()
    assistant_for_healthcheck = NHSAssistant() 

    while True:
        logger.info(f"--- Health Monitor Cycle ({time.ctime()}) ---")
        
        # 1. System Health Check
        health_report_dict = health_checker.check_system_health() 
        logger.info(f"System Health: Overall={health_report_dict.get('overall', 'unknown')}, Warnings={len(health_report_dict.get('warnings',[]))}, Errors={len(health_report_dict.get('errors',[]))}")
        if health_report_dict.get("overall") == "critical":
            logger.critical("!!! ALERT: CRITICAL SYSTEM HEALTH DETECTED !!!")
            audit_logger.critical(f"ALERT_TRIGGERED - Type: SystemHealthCritical, Details: {health_report_dict.get('errors')}")
            print(f"CRITICAL ALERT at {time.ctime()}: System health critical. Check logs. Errors: {health_report_dict.get('errors')}")
        elif health_report_dict.get("overall") == "warning":
            logger.warning("System health status: WARNING. Check logs for details.")
            audit_logger.warning(f"HEALTH_EVENT - Type: SystemHealthWarning, Details: {health_report_dict.get('warnings')}")


        # 2. AI Model Responsiveness Check
        model_responsive = check_ai_model_responsiveness(assistant_for_healthcheck)
        if not model_responsive:
            logger.error("!!! ALERT: AI MODEL IS UNRESPONSIVE OR RETURNING FALLBACKS !!!")
            audit_logger.error("ALERT_TRIGGERED - Type: AIModelUnresponsive, Details: Check main log for AI responsiveness check failure.")
            print(f"CRITICAL ALERT at {time.ctime()}: AI Model unresponsive. Check logs.")

        # 3. Track other performance metrics
        track_performance_metrics(assistant_for_healthcheck)
        
        logger.info(f"--- End Health Monitor Cycle ({time.ctime()}) ---")
        time.sleep(interval_seconds)

if __name__ == "__main__":
    print("NHS Navigator Health Monitor (Simulation - 1 cycle)")
    logger.info("Health Monitor script started directly (simulation mode).")
    audit_logger.info("SYSTEM_EVENT - Type: HealthMonitorStarted, Mode: ManualScriptRun")

    health_checker_test = SystemHealthChecker()
    assistant_test = NHSAssistant() 

    print("\n--- Running System Health Check ---")
    report_output_dict = health_checker_test.check_system_health() # Get dict
    print(f"Overall Health: {report_output_dict.get('overall', 'unknown').upper()}")
    if report_output_dict.get('warnings'): print(f"Warnings: {report_output_dict.get('warnings')}")
    if report_output_dict.get('errors'): print(f"Errors: {report_output_dict.get('errors')}")
    # For more detailed textual report if needed:
    # print(health_checker_test.create_diagnostic_report()) 
    
    print("\n--- Running AI Model Responsiveness Check ---")
    responsive = check_ai_model_responsiveness(assistant_test)
    print(f"AI Model Responsive: {'✅ YES' if responsive else '❌ NO'}")

    print("\n--- Tracking Performance Metrics (Conceptual) ---")
    track_performance_metrics(assistant_test)
    
    print("\nTo run continuously, call run_health_monitoring_loop().")
    logger.info("Health Monitor script direct run finished.")
    audit_logger.info("SYSTEM_EVENT - Type: HealthMonitorFinished, Mode: ManualScriptRun")
