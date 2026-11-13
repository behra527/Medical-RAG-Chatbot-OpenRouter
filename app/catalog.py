"""Verified publisher URLs only. Import is opt-in and subject to licence checks."""
from __future__ import annotations
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Guideline:
    id: str
    title: str
    publisher: str
    year: int
    topic: str
    pdf_url: str
    info_url: str
    license_note: str
    requires_permission: bool = False

    def public(self) -> dict:
        result = asdict(self)
        # Do not present the URL as an endorsement, only as its publisher's link.
        return result

CATALOG = {
    'who-hypertension-2021': Guideline(
        id='who-hypertension-2021',
        title='Guideline for the pharmacological treatment of hypertension in adults',
        publisher='World Health Organization', year=2021, topic='Hypertension',
        pdf_url='https://iris.who.int/bitstream/handle/10665/344424/9789240033986-eng.pdf',
        info_url='https://www.who.int/publications/i/item/9789240033986',
        license_note='CC BY-NC-SA 3.0 IGO: non-commercial use only unless you have WHO permission; attribution required.',
        requires_permission=False,
    ),
    'nice-ng136-2026': Guideline(
        id='nice-ng136-2026',
        title='NG136 — Hypertension in adults: diagnosis and management',
        publisher='National Institute for Health and Care Excellence', year=2026, topic='Hypertension',
        pdf_url='https://www.nice.org.uk/guidance/ng136/resources/hypertension-in-adults-diagnosis-and-managementpdf-66141722710213',
        info_url='https://www.nice.org.uk/guidance/ng136',
        license_note='NICE UK Open Content Licence covers UK-only reuse. Outside UK, contact NICE for reuse permission (beyond personal study/research).',
        requires_permission=True,
    ),
}
