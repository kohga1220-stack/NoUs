"""
Scepter registry — defines all 5 Scepters and their domain identities.
"""
from nous.scepter.base import Scepter


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
