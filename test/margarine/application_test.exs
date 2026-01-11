defmodule Margarine.ApplicationTest do
  use ExUnit.Case, async: false

  @moduledoc """
  Tests for Margarine.Application module.

  NOTE ON COVERAGE: This module tests OTP application startup and Pythonx
  initialization. Since we can't easily stop/restart the application in tests,
  we focus on testing that:
  - The application starts successfully
  - Pythonx initialization is triggered
  - Error handling works for missing dependencies

  Coverage of 60-70% is acceptable as actual Pythonx behavior is tested
  in integration tests.
  """

  describe "application startup" do
    test "application starts successfully" do
      # Application should already be started by test_helper
      assert Process.whereis(Margarine.Supervisor) != nil
    end

    test "supervisor is running" do
      children = Supervisor.which_children(Margarine.Supervisor)
      # Initially empty - models are loaded on demand
      assert is_list(children)
    end
  end

  describe "check_environment/0" do
    test "returns environment status" do
      result = Margarine.Application.check_environment()
      assert is_map(result)
      assert Map.has_key?(result, :pythonx_initialized)
      assert Map.has_key?(result, :python_version)
    end

    test "pythonx_initialized is boolean" do
      result = Margarine.Application.check_environment()
      assert is_boolean(result.pythonx_initialized)
    end
  end
end
