from time import sleep


class Panel:
    Practice = "Practice"
    Real = "Real"


class Mode:
    Invest = "Invest"
    CFD = "CFD"


def force_click(element, sleep_time=1):
    while True:
        try:
            element.click()
            return
        except:
            sleep(sleep_time)


def type_sleep(element, text, time):
    for t in text:
        element.send_keys(t)
        sleep(time)


def script_click_xpath(driver, xpath):
    driver.execute_script(
        f"document.evaluate(\"{xpath}\", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue.click()")
