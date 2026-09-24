"""Offline templated toy worlds.

Purpose: an end-to-end smoke test of the pipeline with zero downloads, and a
tiny controlled probe for multi-hop sufficiency. Not intended to train a real
model; the language is deliberately simple and the entities are random.

Each template returns (question, gold_facts, relevant_distractors, answer,
multi_hop). Relevant distractors are on-topic sentences about the *same*
entities that do not resolve the question, which is what a relevance shortcut
would latch onto.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from ..schema import Example
from .degrade import VariantSpec, build_variants

_COMPANIES = ["Acme", "Globex", "Initech", "Umbrella", "Vandelay", "Hooli", "Stark", "Wayne", "Wonka", "Tyrell"]
_CITIES = {
    "Lyon": "France",
    "Osaka": "Japan",
    "Porto": "Portugal",
    "Austin": "the United States",
    "Leeds": "the United Kingdom",
    "Pune": "India",
    "Bergen": "Norway",
    "Cusco": "Peru",
    "Gdansk": "Poland",
    "Cebu": "the Philippines",
}
_PLANS = ["Free", "Starter", "Team", "Enterprise", "Legacy"]
_PEOPLE = [
    "Mara Lindqvist",
    "Tomas Reyes",
    "Ayesha Khan",
    "Ben Okafor",
    "Ines Moreau",
    "Kenji Sato",
    "Lena Fischer",
    "Rahul Menon",
]
_DEVICES = ["DermaPatch", "CardioLink", "NeuroBand", "OrthoFix", "OptiLens", "PulseTag"]
_WAREHOUSES = ["Reno", "Memphis", "Rotterdam", "Chennai", "Melbourne"]
_REGIONS = ["the Northeast", "the Pacific coast", "Central Europe", "South Asia", "Oceania"]

_UNRELATED = [
    "The quarterly newsletter will be published next Tuesday.",
    "Parking on the north lot is reserved for visitors.",
    "The cafeteria now serves oat milk on request.",
    "A software update was rolled out to all laptops last month.",
    "The annual offsite will take place in October.",
    "The office thermostat is set to 21 degrees.",
    "New badge readers were installed at the east entrance.",
    "The library extended its weekend opening hours.",
    "The bridge repainting project finished ahead of schedule.",
    "A local bakery won a regional award for its sourdough.",
    "The museum added an exhibit on early printing presses.",
    "Rainfall this spring was slightly above the long-term average.",
]


def _pick(rng: random.Random, xs: list[str]) -> str:
    return rng.choice(xs)


def t_plan_access(rng: random.Random):
    name = _pick(rng, _PEOPLE)
    plan = _pick(rng, _PLANS)
    allowed = rng.random() < 0.5
    other_plan = _pick(rng, [p for p in _PLANS if p != plan])
    gold = [
        f"{name} is on the {plan} plan.",
        f"{plan} plans {'permit' if allowed else 'do not permit'} API access.",
    ]
    distract = [
        f"{name} created their account in {rng.randint(2015, 2024)}.",
        f"{other_plan} plans {'permit' if not allowed else 'do not permit'} API access.",
        f"{name} recently changed their billing email.",
        f"The {plan} plan is billed {'monthly' if rng.random() < 0.5 else 'annually'}.",
        "API access can be requested through the developer portal.",
        f"{name} attended the onboarding webinar.",
    ]
    q = f"Does {name} have API access?"
    return q, gold, distract, "yes" if allowed else "no", True


def t_hq_country(rng: random.Random):
    company = _pick(rng, _COMPANIES)
    city = _pick(rng, list(_CITIES))
    country = _CITIES[city]
    other_city = _pick(rng, [c for c in _CITIES if c != city])
    gold = [
        f"{company} is headquartered in {city}.",
        f"{city} is a city in {country}.",
    ]
    distract = [
        f"{company} was founded in {rng.randint(1950, 2015)}.",
        f"{company} opened a sales office in {other_city} last year.",
        f"{other_city} is a city in {_CITIES[other_city]}.",
        f"{company} employs about {rng.randint(2, 90) * 100} people.",
        f"{city} hosts an annual technology conference.",
        f"{company}'s chief executive spoke at an industry event.",
    ]
    q = f"In which country is {company} headquartered?"
    return q, gold, distract, country, True


def t_device_testing(rng: random.Random):
    device = _pick(rng, _DEVICES)
    days = rng.choice([3, 7, 14, 21, 29, 31, 45, 60, 90, 180])
    threshold = 30
    gold = [
        f"The {device} has a contact duration of {days} days.",
        f"Devices with a contact duration over {threshold} days require biocompatibility testing.",
    ]
    distract = [
        f"The {device} is made of {rng.choice(['silicone', 'polyurethane', 'stainless steel', 'titanium'])}.",
        f"The {device} contacts intact skin.",
        f"The {device} was first marketed in {rng.randint(2005, 2023)}.",
        "Biocompatibility testing typically includes cytotoxicity and sensitization assays.",
        f"The {device} is distributed in {rng.randint(3, 40)} countries.",
        "The predicate device underwent irritation testing.",
    ]
    q = f"Does the {device} require biocompatibility testing?"
    return q, gold, distract, "yes" if days > threshold else "no", True


def t_birth_year(rng: random.Random):
    name = _pick(rng, _PEOPLE)
    year = rng.randint(1940, 2000)
    gold = [f"{name} was born in {year}."]
    distract = [
        f"{name} studied {rng.choice(['chemistry', 'law', 'architecture', 'history'])} at university.",
        f"{name} moved to {_pick(rng, list(_CITIES))} in {year + rng.randint(18, 30)}.",
        f"{name} has published {rng.randint(1, 12)} books.",
        f"{name} received an award for community service.",
    ]
    q = f"In what year was {name} born?"
    return q, gold, distract, str(year), False


def t_delivery(rng: random.Random):
    order = f"Order #{rng.randint(10000, 99999)}"
    wh = _pick(rng, _WAREHOUSES)
    region = _pick(rng, _REGIONS)
    days = rng.randint(2, 9)
    ship_day = rng.randint(1, 20)
    gold = [
        f"{order} shipped from the {wh} warehouse on March {ship_day}.",
        f"{order} is being delivered to {region}.",
        f"Shipments from {wh} to {region} take {days} days.",
    ]
    distract = [
        f"{order} contains {rng.randint(1, 6)} items.",
        f"{order} was paid by {rng.choice(['credit card', 'invoice', 'bank transfer'])}.",
        f"The {wh} warehouse operates six days a week.",
        f"Shipments from {_pick(rng, [w for w in _WAREHOUSES if w != wh])} to {region} take {days + 2} days.",
        f"{order} was placed on March {max(1, ship_day - rng.randint(1, 3))}.",
        "Customers receive a tracking link by email.",
    ]
    q = f"On what date will {order} arrive?"
    return q, gold, distract, f"March {ship_day + days}", True


TEMPLATES: list[Callable[[random.Random], tuple]] = [
    t_plan_access,
    t_hq_country,
    t_device_testing,
    t_birth_year,
    t_delivery,
]


def generate_toy(n_questions: int, seed: int = 0, spec: VariantSpec | None = None) -> list[Example]:
    rng = random.Random(seed)
    spec = spec or VariantSpec(
        n_clean_sufficient=1,
        n_distractor_heavy=1,
        n_missing_one=1,
        n_relevant_insufficient=1,
        n_long_insufficient=0,
        n_obvious_insufficient=0,
        light_noise_max=2,
        heavy_noise_min=3,
        heavy_noise_max=6,
    )
    out: list[Example] = []
    for i in range(n_questions):
        template = TEMPLATES[i % len(TEMPLATES)]
        q, gold, distract, answer, multi_hop = template(rng)
        # Occasionally emit an obvious_insufficient variant using unrelated noise.
        s = spec
        if i % 7 == 0:
            s = VariantSpec(**{**spec.__dict__, "n_obvious_insufficient": 1})
        out.extend(
            build_variants(
                q,
                gold,
                distract,
                _UNRELATED,
                source="toy",
                group_id=f"toy-{template.__name__}-{i}",
                rng=rng,
                spec=s,
                answer=answer,
                multi_hop=multi_hop,
            )
        )
    return out
