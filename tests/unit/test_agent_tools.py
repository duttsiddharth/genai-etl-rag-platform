from src.genai.agents.tools import CalculatorTool, WebLookupTool


def test_calculator_evaluates_basic_expression():
    tool = CalculatorTool()
    assert tool.run({"expression": "2 + 3 * 4"}) == "14"


def test_calculator_handles_parentheses_and_division():
    tool = CalculatorTool()
    assert tool.run({"expression": "(10 - 4) / 2"}) == "3.0"


def test_calculator_rejects_unsafe_expressions():
    tool = CalculatorTool()
    result = tool.run({"expression": "__import__('os').system('echo hacked')"})
    assert "calculation_error" in result


def test_web_lookup_tool_returns_stub_label():
    tool = WebLookupTool()
    result = tool.run({"query": "latest AWS Bedrock pricing"})
    assert "[stub]" in result
    assert "latest AWS Bedrock pricing" in result
