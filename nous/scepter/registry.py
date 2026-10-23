"""
Scepter registry.

  ALL_SCEPTERS      the 5 parent Scepters (H S N A I). Also the 5 persona judges of `evaluate`.
  FIELD_SCEPTERS    10 field Scepters, the generators of a council debate (children of the 5).
  AUDITORS          10 role-based auditors, each checks one aspect of the top hypotheses.

Adding a Scepter or an auditor = adding a row to FIELD_TABLE / AUDITOR_TABLE (plus its fields in
`nous.domains.FIELD_GROUPS` for a field Scepter).
"""
from nous.domains import FIELD_GROUPS
from nous.scepter.base import Auditor, Scepter


class ScepterH(Scepter):
    name = "Scepter-H"
    domain = "Humanities"
    subdomains = [
        "Literature", "Philosophy", "Ethics", "Religious Studies",
        "History", "Geography", "Archaeology",
        "Psychology", "Cultural Anthropology",
        "Linguistics", "Language Studies",
    ]
    lens = (
        "Interprets phenomena through the lens of meaning, narrative, "
        "cultural memory, and the lived human experience across time and place."
    )


class ScepterS(Scepter):
    name = "Scepter-S"
    domain = "Social Sciences"
    subdomains = [
        "Law", "Political Science", "International Relations",
        "Economics", "Management", "Accounting", "Marketing",
        "Sociology", "Media Studies", "Tourism",
        "Education", "Lifelong Learning",
    ]
    lens = (
        "Analyzes structural power, institutional dynamics, incentive systems, "
        "and collective human behavior at societal and organizational scales."
    )


class ScepterN(Scepter):
    name = "Scepter-N"
    domain = "Natural Sciences"
    subdomains = [
        "Algebra", "Geometry", "Analysis", "Statistics",
        "Particle Physics", "Astrophysics", "Condensed Matter Physics",
        "Organic Chemistry", "Inorganic Chemistry", "Physical Chemistry",
        "Molecular Biology", "Ecology", "Cell Biology",
        "Geology", "Meteorology", "Astronomy",
    ]
    lens = (
        "Seeks universal laws, mathematical invariants, and empirically "
        "falsifiable models that describe the physical and biological world."
    )


class ScepterA(Scepter):
    name = "Scepter-A"
    domain = "Applied Sciences"
    subdomains = [
        "Mechanical Engineering", "Electrical Engineering",
        "Information & Communication Technology", "Civil Engineering", "Architecture",
        "Agricultural Chemistry", "Forest Science", "Aquatic Life Sciences",
        "Medicine", "Dentistry", "Pharmacy", "Nursing", "Physical Therapy",
        "Computer Science", "AI Research", "Data Science",
    ]
    lens = (
        "Translates theoretical principles into functional systems, "
        "optimizing for real-world performance, safety, and scalability."
    )


class ScepterI(Scepter):
    name = "Scepter-I"
    domain = "Interdisciplinary & Human Sciences"
    subdomains = [
        "Sports Science", "Sports Medicine", "Health & Welfare",
        "International Studies", "Environmental Science", "Disaster Prevention",
        "Human Relations", "Social Welfare",
    ]
    lens = (
        "Identifies emergent patterns at the intersection of multiple disciplines, "
        "focusing on human flourishing, systemic risk, and cross-domain synthesis."
    )


# Instantiate all Scepters
ALL_SCEPTERS: list[Scepter] = [
    ScepterH(), ScepterS(), ScepterN(), ScepterA(), ScepterI()
]

SCEPTER_MAP: dict[str, Scepter] = {s.name: s for s in ALL_SCEPTERS}


# ------------------------------------------------------------------ #
#  Table-driven council: 10 field Scepters + 10 auditors              #
# ------------------------------------------------------------------ #

# name -> (domain label, sub-domains, lens)
FIELD_TABLE: dict[str, tuple[str, list[str], str]] = {
    "Scepter-H1": ("Arts & Humanities",
                   ["Literature", "Philosophy", "Ethics", "Religious Studies", "History",
                    "Archaeology", "Linguistics", "Language Studies"],
                   "Reads phenomena as meaning, narrative and cultural memory across time and place."),
    "Scepter-H2": ("Psychology",
                   ["Cognitive Psychology", "Developmental Psychology", "Social Psychology",
                    "Clinical Psychology", "Psychometrics"],
                   "Explains behaviour through cognition, development, emotion and measurable individual differences."),
    "Scepter-S1": ("Social Sciences",
                   ["Sociology", "Law", "Political Science", "Education", "Media Studies",
                    "Anthropology", "Human Geography"],
                   "Analyses institutions, norms, power and collective behaviour at the scale of societies."),
    "Scepter-S2": ("Economics, Management & Decision Sciences",
                   ["Economics", "Finance", "Management", "Accounting", "Marketing",
                    "Operations Research", "Game Theory"],
                   "Reasons with incentives, constraints, markets and optimal choice under uncertainty."),
    "Scepter-N1": ("Mathematics & Physics",
                   ["Algebra", "Geometry", "Analysis", "Statistics", "Particle Physics",
                    "Astrophysics", "Condensed Matter Physics"],
                   "Looks for invariants, symmetries and exact laws that hold regardless of the particular system."),
    "Scepter-N2": ("Chemistry & Earth Sciences",
                   ["Organic Chemistry", "Inorganic Chemistry", "Physical Chemistry",
                    "Geology", "Meteorology", "Oceanography"],
                   "Thinks in terms of reactions, transport, equilibria and the long timescales of the planet."),
    "Scepter-N3": ("Life Sciences & Neuroscience",
                   ["Molecular Biology", "Genetics", "Cell Biology", "Ecology", "Microbiology",
                    "Immunology", "Neuroscience"],
                   "Explains living systems through evolution, regulation, networks of cells and neural circuits."),
    "Scepter-A1": ("Computing & Engineering",
                   ["Computer Science", "AI Research", "Data Science", "Electrical Engineering",
                    "Mechanical Engineering", "Materials Science", "Energy Systems"],
                   "Turns principles into working systems, trading off performance, robustness and cost."),
    "Scepter-A2": ("Clinical & Health Sciences",
                   ["Medicine", "Dentistry", "Nursing", "Pharmacology", "Public Health",
                    "Rehabilitation", "Veterinary Medicine"],
                   "Judges ideas by clinical evidence, patient outcomes, dose-response and harm."),
    "Scepter-I1": ("Environment & Systems",
                   ["Environmental Science", "Sustainability", "Disaster Prevention",
                    "Systems Ecology", "Climate Adaptation"],
                   "Looks for emergent patterns and systemic risk where several disciplines and scales meet."),
}

# name -> (role label, what it checks, the one question it answers)
AUDITOR_TABLE: dict[str, tuple[str, str, str]] = {
    "Auditor-PriorArt": ("Prior art",
                         "Knows what has already been published.",
                         "Is this claim already known, under this or another name?"),
    "Auditor-Triviality": ("Triviality",
                           "Spots claims that are true but say nothing new.",
                           "Would an expert find this obvious or true by definition?"),
    "Auditor-Coherence": ("Coherence & ontology fit",
                          "Checks that the concepts joined by the hypothesis can legitimately be joined.",
                          "Do the terms mean the same thing on both sides, and are they the right kind of thing to relate?"),
    "Auditor-Measurability": ("Measurability",
                              "Checks that every key term can be operationalised.",
                              "Can each central quantity be measured with a real instrument or dataset?"),
    "Auditor-Statistics": ("Statistics",
                           "Checks that the proposed test can actually separate the claim from chance.",
                           "Would the proposed analysis have the power and design to tell the claim from noise or confounding?"),
    "Auditor-Falsifier": ("Falsifiability",
                          "Looks for the result that would refute the hypothesis.",
                          "What observation would show this is wrong, and is such an observation possible?"),
    "Auditor-Ethics": ("Ethics & risk",
                       "Checks harms to people, misuse and consent.",
                       "Could testing or applying this harm someone, or be misused?"),
    "Auditor-Feasibility": ("Feasibility",
                            "Checks time, data, cost and access.",
                            "Could a small team test this within a year with data that exists or is obtainable?"),
    "Auditor-History": ("History of science",
                        "Knows how similar claims fared before.",
                        "Has this kind of cross-field claim been tried and abandoned, and why?"),
    "Auditor-Terminology": ("Terminology translation",
                            "Translates between the vocabularies of the fields involved.",
                            "Is the same word used for different things (or different words for the same thing) across the fields?"),
}


def _make_field(name: str, parent: str, spec: tuple) -> type[Scepter]:
    domain, subdomains, lens = spec
    return type(name.replace("-", ""), (Scepter,),
                {"name": name, "domain": domain, "subdomains": subdomains, "lens": lens,
                 "parent": parent})


def _make_auditor(name: str, spec: tuple) -> type[Auditor]:
    domain, role, question = spec
    return type(name.replace("-", ""), (Auditor,),
                {"name": name, "domain": domain, "role": role, "question": question})


FIELD_SCEPTERS: list[Scepter] = [
    _make_field(name, parent, FIELD_TABLE[name])()
    for name, parent, _fields in FIELD_GROUPS
]
AUDITORS: list[Auditor] = [_make_auditor(n, spec)() for n, spec in AUDITOR_TABLE.items()]

COUNCIL_MAP: dict[str, object] = {a.name: a for a in [*ALL_SCEPTERS, *FIELD_SCEPTERS, *AUDITORS]}
