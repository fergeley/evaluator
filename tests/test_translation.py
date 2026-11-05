import sys
import os
import time
import pytest
from statistics import mean, stdev
from unittest.mock import Mock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import LLMConfig
from services.malay2sql_service import Malay2SQLService

# Translation runs on the local nanot5 model and never calls the LLM,
# so the service gets a placeholder LLM config here.
_service = None


def initialize_translator() -> Malay2SQLService:
    global _service
    if _service is None:
        _service = Malay2SQLService(
            LLMConfig(provider="openai", api_key="unused", model="unused")
        )
    return _service


def translate_malay_to_english(text):
    return initialize_translator().translate_malay_to_english(text)

# Unit Tests
def test_translator_initialization():
    """Unit test to verify translator initialization"""
    translator = initialize_translator()
    assert translator is not None
    assert hasattr(translator, 'translate_malay_to_english')

def test_invalid_input():
    """Unit test for invalid inputs"""
    test_cases = [
        None,
        "",
        " ",
        123,  # non-string input
        "a" * 1000  # very long input
    ]
    
    for test_input in test_cases:
        try:
            result = translate_malay_to_english(test_input)
            assert result is not None, f"Translation should handle {test_input}"
        except Exception as e:
            print(f"Error handling {test_input}: {str(e)}")

# Integration Tests
def test_translation_quality():
    """Integration test for translation quality"""
    test_cases = [
        {
            "input": "Siapa nama kamu?",
            "expected_contains": ["what", "name", "your"],
            "should_not_contain": ["siapa", "nama"]
        },
        {
            "input": "Berapa harga barang ini?",
            "expected_contains": ["how", "much", "price"],
            "should_not_contain": ["berapa", "harga"]
        },
        {
            "input": "ko memang cam babi sial",
            "expected_contains": ["you"],
            "should_not_contain": ["ko", "babi"]
        }
    ]
    
    for test_case in test_cases:
        translation = translate_malay_to_english(test_case["input"]).lower()
        
        # Check if translation contains expected words
        for expected_word in test_case["expected_contains"]:
            assert expected_word in translation, f"Translation should contain '{expected_word}'"
        
        # Check if translation doesn't contain unwanted words
        for unwanted_word in test_case["should_not_contain"]:
            assert unwanted_word not in translation, f"Translation shouldn't contain '{unwanted_word}'"

# System Tests
def test_end_to_end_performance():
    """System test for end-to-end performance and reliability"""
    iterations = 5
    max_allowed_time = 5000  # 5 seconds in milliseconds
    
    test_cases = [
        "ko memang cam babi sial",
        "Siapa nama kamu?",
        "Berapa harga barang ini?",
        "Saya suka makan nasi goreng",
        "Bila anda mahu pergi ke kedai?",
    ]

    print("\nSystem Testing - Performance and Reliability:")
    print("-" * 50)
    
    results = []
    errors = 0
    
    for malay_text in test_cases:
        times = []
        success = 0
        
        for _ in range(iterations):
            try:
                start_time = time.time()
                translation = translate_malay_to_english(malay_text)
                end_time = time.time()
                
                elapsed_time = (end_time - start_time) * 1000
                times.append(elapsed_time)
                
                if translation and isinstance(translation, str):
                    success += 1
                
                assert elapsed_time < max_allowed_time, f"Translation took too long: {elapsed_time}ms"
                
            except Exception as e:
                errors += 1
                print(f"Error processing '{malay_text}': {str(e)}")
        
        avg_time = mean(times)
        std_dev = stdev(times) if len(times) > 1 else 0
        success_rate = (success / iterations) * 100
        
        results.append({
            'text': malay_text,
            'avg_time': avg_time,
            'std_dev': std_dev,
            'success_rate': success_rate
        })
        
        print(f"\nTest case: {malay_text}")
        print(f"Average time: {avg_time:.2f}ms (±{std_dev:.2f}ms)")
        print(f"Success rate: {success_rate}%")

    # Overall system metrics
    total_success_rate = mean([r['success_rate'] for r in results])
    avg_response_time = mean([r['avg_time'] for r in results])
    
    print("\nOverall System Metrics:")
    print("-" * 50)
    print(f"Total test cases: {len(test_cases)}")
    print(f"Iterations per test: {iterations}")
    print(f"Overall success rate: {total_success_rate:.2f}%")
    print(f"Average response time: {avg_response_time:.2f}ms")
    print(f"Total errors: {errors}")
    
    # System test assertions
    assert total_success_rate >= 95, "System success rate below 95%"
    assert avg_response_time < max_allowed_time, "Average response time too high"
    assert errors == 0, "System test encountered errors"

if __name__ == "__main__":
    # Print GPU information
    print("\nGPU Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Run all tests
    pytest.main([__file__])