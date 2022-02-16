from hover.utils.bokeh_helper import binder_proxy_app_url, remote_jupyter_proxy_url
from urllib.parse import urlparse
import pytest


@pytest.mark.lite
def test_binder_proxy_app_url():
    """
    The function being tested is only intended for Binder.
    """
    url = binder_proxy_app_url("simple-annotator", port=5007)
    _ = urlparse(url)


@pytest.mark.lite
def test_remote_jupyter_proxy_url():
    """
    Not a full test, rather just validating urls.
    """
    url = remote_jupyter_proxy_url(8888)
    _ = urlparse(url)