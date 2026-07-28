import sys
from src.logger import logging


def error_message_detail(error, error_detail: sys):
    ## Get information about the most recent exception
    _, _, exc_tb = error_detail.exc_info()
    
    ## Get the filename where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    ## Create a detailed error message
    error_message = (
        "Error occurred in python script name "
        f"[{file_name}]"
        f"line number [{exc_tb.tb_lineno}]"
        f"error message [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        ## Initialize the parent exception class
        super().__init__(error_message)
        
        ## store our custom detailed error message
        self.error_message = error_message_detail(
            error_message,
            error_detail
        )
        
    def __str__(self):
        ## Return the custom error message when the exception is printed
        return self.error_message