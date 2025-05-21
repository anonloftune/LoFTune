from datetime import datetime

def decorate_str_with_date(value: str):
    """
    Add current datetime to a given string value

    :param value: a string
    :return: Value decorated with current datetime: value_%d_%m_%Y_%H_%M_%S
    """
    now = datetime.now()
    now = now.strftime("%d_%m_%Y_%H_%M_%S")
    return f"{value}_{now}"
