from alpha_vantage.alphavantage import AlphaVantage as av
from functools import wraps


def format_income_statement(func):
    @wraps(func)
    def _format_wrapper(self, *args, **kwargs):
        call_response, data_key, meta_data_key = func(
            self, *args)

        return call_response, None

    return _format_wrapper


class FundamentalData(av):

    @format_income_statement
    @av._call_api_on_func
    def get_income_statement(self, symbol):
        _FUNCTION_KEY = "INCOME_STATEMENT"
        return _FUNCTION_KEY, 'INCOME_STATEMENT', 'Meta Data'

    @format
    @av._call_api_on_func
    def get_income_statement(self, symbol):
        _FUNCTION_KEY = "BALANCE_SHEET"
        return _FUNCTION_KEY, 'BALANCE_SHEET', 'Meta Data'

    @format
    @av._call_api_on_func
    def get_income_statement(self, symbol):
        _FUNCTION_KEY = "CASH_FLOW"
        return _FUNCTION_KEY, 'CASH_FLOW', 'Meta Data'

    @format
    @av._call_api_on_func
    def get_company_overview(self, symbol):
        _FUNCTION_KEY = "CASH_FLOW"
        return _FUNCTION_KEY, 'CASH_FLOW', 'Meta Data'
