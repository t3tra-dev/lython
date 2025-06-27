#!/usr/bin/env python3
"""
Lython テストランナー

全てのテストを実行し、結果を報告します。
"""

import sys
import unittest
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def discover_and_run_tests():
    """テストを発見して実行"""
    print("=" * 60)
    print("Lython TypedAST Implementation Test Suite")
    print("=" * 60)

    # テストローダーを作成
    loader = unittest.TestLoader()

    # 各テストモジュールを個別に実行
    test_modules = [
        "frontend.test_ast_nodes",
        "typing.test_types",
        "native.test_native_api",
        "integration.test_typed_ast_integration",
    ]

    total_tests = 0
    total_failures = 0
    total_errors = 0

    for module_name in test_modules:
        print(f"\n--- Running {module_name} ---")

        try:
            # テストスイートを作成
            suite = loader.loadTestsFromName(f"tests.{module_name}")

            # テストランナーを作成
            runner = unittest.TextTestRunner(
                verbosity=2, stream=sys.stdout, buffer=True
            )

            # テストを実行
            result = runner.run(suite)

            # 結果を集計
            total_tests += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)

        except Exception as e:
            print(f"Error loading tests from {module_name}: {e}")
            total_errors += 1

    # 最終結果の表示
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")

    if total_failures == 0 and total_errors == 0:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


def run_specific_test_module(module_name):
    """特定のテストモジュールを実行"""
    print(f"Running tests from {module_name}")

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(f"tests.{module_name}")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


def main():
    """メイン関数"""
    if len(sys.argv) > 1:
        # 特定のモジュールを指定
        module_name = sys.argv[1]
        return run_specific_test_module(module_name)
    else:
        # 全テストを実行
        return discover_and_run_tests()


if __name__ == "__main__":
    sys.exit(main())
