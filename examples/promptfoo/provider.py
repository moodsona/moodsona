"""The file promptfoo loads as a Python provider: ``id: file://provider.py``.

Copy it next to your ``promptfooconfig.yaml``. It only re-exports moodsona's
provider function, so the Python that promptfoo runs must have moodsona
installed. See docs/integrations/promptfoo.md.
"""

from moodsona.adapters.promptfoo import call_api

__all__ = ["call_api"]
