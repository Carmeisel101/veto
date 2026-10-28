"""Synthetic message dataset for running Veto without private data.

Everything here is invented: names, phone numbers (reserved 555-01XX style), URLs (``.example``
domain), organizations and brands. No real message or person is used. Positive messages are
political solicitations in the styles Veto targets; negatives include personal chat, hard
negatives (friends talking politics, news alerts, commercial promos using political words) and
non-political spam. Labels are known by construction.

The output matches the raw-export schema (see ``docs/raw_extraction.md``) plus an annotations
table in the format read by ``veto.dataset.labels``.
"""

from __future__ import annotations

import random

import pandas as pd

SYNTHETIC_LABEL_SOURCE = "synthetic"

# Base composition for n=1000 (scaled for other sizes; remainder goes to personal chat).
_BASE_COUNTS = {
    "campaign": 33, "fundraising": 25, "advocacy": 12, "election_information": 12,
    "polling_or_survey": 10, "petition": 4, "political_event": 4,
    "political_talk": 30, "news_alert": 10, "transactional": 90, "commercial": 100,
    "nonpolitical_spam": 40, "other": 30,
    "review_contentless": 15, "review_ambiguous": 5,
}
_POSITIVES = ("campaign", "fundraising", "advocacy", "election_information",
              "polling_or_survey", "petition", "political_event")

_FIRST = ["Dana", "Marcus", "Priya", "Tomas", "Elena", "Jamal", "Noor", "Wes", "Ingrid", "Rafael",
          "Hana", "Cole", "Mira", "Dev", "Sasha", "Theo", "Lena", "Omar", "Rhea", "Gus",
          "Tessa", "Boyd", "Nia", "Felix", "Ada", "Kit", "Lucas", "Maya", "Reid", "Zoe"]
_LAST = ["Whitfield", "Okafor", "Lindqvist", "Marchetti", "Haddad", "Brennan", "Castellano",
         "Voss", "Abernathy", "Takahashi", "Delgado", "Fairchild", "Nakamura", "Oyelaran",
         "Pemberton", "Quint", "Rosales", "Sterling", "Thackeray", "Underhill", "Vance",
         "Wexler", "Yarborough", "Zeller", "Ashworth", "Bellamy", "Crowe", "Dunmore"]
_OFFICES = ["Governor", "U.S. Senate", "Mayor", "County Judge", "State Representative",
            "County Commissioner", "School Board", "Attorney General"]
_ORGS = ["Citizens for Safe Streets", "Alliance for Public Schools", "Clean Water Now Coalition",
         "Taxpayers United Fund", "Rural Voices Network", "Neighbors for Fair Housing",
         "Workers First Alliance", "Open Government Project"]
_ISSUES = ["clean water", "school funding", "property tax relief", "public safety",
           "affordable housing", "healthcare access", "ethics reform", "local road repair"]
_POLLSTERS = ["Meridian Opinion Lab", "Prairie Research Group", "Civic Pulse Surveys",
              "Northgate Polling"]
_VENUES = ["the Grange Hall", "Riverside Park pavilion", "Lincoln High gym", "the Union Hall",
           "Maple Street Community Center"]
_BRANDS = ["Boltwear", "FreshCrate", "Nimbus Gear", "Harvest & Co", "TrailForge", "PetPals",
           "Glowhaus", "Sunnyside Coffee", "Pixel Pantry", "Orbit Outfitters"]
_FOODS = ["tacos", "pizza", "sushi", "burgers", "pho", "wings", "ramen", "bagels"]
_PLACES = ["the usual spot", "that new place downtown", "your place", "the lake", "the gym",
           "campus", "the brewery"]
_MONTHS = ["Oct. 19", "Oct. 24", "Nov. 3", "Nov. 5", "Nov. 10"]
_URL_CHARS = "abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"

_SHORT_PERSONAL = [
    "lol", "omg", "ok", "yes", "no way", "on my way", "be there in 10", "what time?", "thanks!",
    "sounds good", "haha same", "can you call me?", "just got home", "running late", "love you",
    "good morning 🩷", "good night!", "what are you up to", "did you eat yet?", "k", "bet",
    "nice!!", "😂😂", "I'm so tired", "see you tomorrow", "wait what", "no worries", "you up?",
    "that's crazy", "heading out now", "👍", "congrats!!", "happy birthday!!", "miss you",
    "can't wait", "I'll text you later", "yeah I saw that", "she said yes!", "ugh Mondays",
]
_PERSONAL_TEMPLATES = [
    "Want to grab {food} at {place}?", "I'm gonna be {n} min late", "Did you see the game last night?",
    "Can you pick up {food} on the way home?", "Mom says dinner is at {n}", "Just landed, call you in a bit",
    "Are we still on for Saturday?", "Don't forget {name}'s party on Friday", "Who's driving tomorrow?",
    "I think I left my charger at {place}", "{name} just texted me, he's coming too", "Gym at {n}?",
    "Can you send me that photo from {place}?", "The wifi is down again 😩", "Just finished a {n} mile run",
    "I can't believe it's already {month}", "Thanks for helping me move yesterday!",
    "Is {name} coming to {place}?", "I'll bring the snacks, you bring the speaker",
]
_POLITICAL_TALK = [
    "Did you watch the debate last night? I thought {cand} did pretty well",
    "Did you vote yet?", "Ugh this election season is exhausting",
    "my uncle won't stop posting about {cand} on facebook",
    "Are you going to vote early or on election day?",
    "I can't believe {opp} said that in the debate lol",
    "We should talk about the ballot propositions before we vote",
    "Honestly I don't like either candidate for {office}",
    "My coworker and I argued about the election all lunch",
    "I'm voting for {cand}, but don't tell my dad 😂",
    "Registered to vote today finally",
    "The line at the polling place was so long this morning",
    "Who are you voting for on the school board thing?",
]
_NEWS = [
    "News alert: {cand} leads {opp} in a new statewide poll, per Metro Tribune.",
    "Metro Tribune: {cand} and {opp} meet for the first televised {office} debate tonight at 7.",
    "BREAKING: Polls open at 7am on Nov. 3. Check Metro Tribune for local results all night.",
    "Channel 6 News: Early voting turnout up {n}% compared to last cycle.",
    "Alert: {opp} concedes the {office} race, Metro Tribune reports.",
]
_TRANSACTIONAL = [
    "Your {brand} verification code is {code}. It expires in 10 minutes. Don't share it.",
    "{code} is your login code. Never share it with anyone.",
    "Reminder: your appointment at Greenleaf Dental is {month} at {n}:30 PM. Reply C to confirm.",
    "Your order has shipped! Track it here: {url}",
    "Your package will arrive today between {n}am and 5pm. Track: {url}",
    "Fraud alert: did you attempt a ${amt} purchase at an unfamiliar merchant? Reply YES or NO.",
    "It's time to check in for your flight. Confirmation: {code}. Reply LANGUAGE to change language.",
    "Your ride is arriving in {n} minutes.",
    "Payment of ${amt}.00 received. Thank you! Your receipt: {url}",
    "Reminder: your car service appointment is tomorrow at {n}:00 AM.",
    "Your prescription is ready for pickup at Elmwood Pharmacy.",
    "Your account balance is below $50. Log in to review.",
]
_COMMERCIAL = [
    "{brand}: NEW ARRIVALS are live. Shop now: {url} Reply STOP to opt out.",
    "{brand}: {n}0% off sitewide through Sunday. Use code SAVE{n}. {url}",
    "{brand} Flash Sale! Everything must go. Ends tonight at midnight: {url}",
    "Vote for your favorite flavor! The winner joins the {brand} lineup. Cast your vote: {url}",
    "{brand}: Election Day Sale: {n}0% off all outerwear. {url} Text STOP to cancel.",
    "Don't miss out! {brand} members get early access to the summer drop. {url}",
    "{brand}: Your cart is waiting. Complete your order and get free shipping: {url}",
    "Join the {brand} team! Vote YES for 2-for-1 Tuesdays. {url}",
    "{brand}: last chance for {n}0% off before the sale ends. {url}",
]
_SPAM = [
    "Hello, this is {name} with {brand} Remote Recruitment. Earn ${amt}-${amt}00 a day working {n} hours from home. Text {phone} for details.",
    "Your toll account is past due. Pay ${n} now to avoid a late fee: {url}",
    "We could not deliver your package. Update your address here: {url}",
    "Exclusive Investor Access: join our private AI wealth program. Limited enrollment: {url}",
    "Congratulations! You've been selected for a ${amt}00 gift card. Claim it: {url}",
    "Your account has been suspended. Verify your identity immediately: {url}",
    "Hi, quick question: would you consider selling your {n} acres? Reply for a cash offer.",
]
_OTHER = [
    "Hi {name}, this is {name2} from Talent Partners. I saw your profile and have a data role that might fit. Do you have time for a call this week?",
    "Reminder: {brand} High School Booster Club fundraiser car wash is Saturday. Donate here to support our team: {url}",
    "Hello from Grace Fellowship! Service is at 10am this Sunday. All are welcome.",
    "Neighborhood watch: a package thief was spotted on Oak Lane. Please lock your doors.",
    "Hi {name}, thanks for your interest! This is {name2}, the owner. Are you still looking at venues for your event?",
    "Help our school raise money for new library books! Every donation counts: {url}",
    "Your HOA meeting is {month} at 7pm in the clubhouse.",
]


class _Gen:
    def __init__(self, seed: int) -> None:
        self.r = random.Random(seed)

    def name(self) -> str:
        return self.r.choice(_FIRST)

    def person(self) -> str:
        return f"{self.r.choice(_FIRST)} {self.r.choice(_LAST)}"

    def url(self, host: str | None = None) -> str:
        host = host or self.r.choice(["vote-now", "give-today", "act-now", "join-us", "my-plan"])
        slug = "".join(self.r.choice(_URL_CHARS) for _ in range(6))
        return f"{host}.example/{slug}"

    def phone(self) -> str:
        return f"+1{self.r.choice([212, 303, 404, 512, 615, 702, 808, 917])}5550{self.r.randint(100, 199)}"

    def fill(self, template: str, **extra) -> str:
        r = self.r
        cand, opp = self.person(), self.person()
        values = {
            "cand": cand, "opp": opp, "office": r.choice(_OFFICES), "url": self.url(),
            "amt": r.choice([5, 10, 15, 20, 25, 50]), "n": r.randint(2, 9), "pct": r.randint(8, 60),
            "month": r.choice(_MONTHS), "name": self.name(), "name2": self.name(),
            "org": r.choice(_ORGS), "issue": r.choice(_ISSUES), "venue": r.choice(_VENUES),
            "pollster": r.choice(_POLLSTERS), "brand": r.choice(_BRANDS), "food": r.choice(_FOODS),
            "place": r.choice(_PLACES), "code": r.randint(100000, 999999), "phone": self.phone(),
            "a": r.randint(44, 52), "b": r.randint(44, 52), "date": r.choice(_MONTHS),
            "rep": self.person(), "year": 2026, "c1": self.person(), "c2": self.person(),
            "vol": self.name(),
        }
        values.update(extra)
        return template.format(**values)


def _positive_texts(g: _Gen, subtype: str) -> str:
    cand = g.person()
    party = g.r.choice("DRI")
    t = {
        "campaign": [
            "{cand} here. {opp} has spent years raising your taxes. Don't let them win in November. Make your plan to vote early: {url}\nReply STOP to end",
            "{cand} for {office}: safer neighborhoods, better schools, lower costs. Early voting starts {date}. Learn more: {url}",
            "Hi {name}, this is {vol}, a volunteer for {cand}'s campaign. Can we count on your vote for {cand} for {office}? Reply YES or NO.",
            "{opp} voted to raise property taxes {n} times. That's not what our community needs. DEFEAT {opp} on Nov. 3. {url}\nStop2End",
            "BREAKING: {cand} just unveiled a plan to cut costs for working families. See it here: {url}",
            "It's up to YOU to stop {opp}. Make your plan to vote ASAP >> {url}\nReply STOP to opt out",
            "{opp} wants to defund our police. We can't let that happen. Vote {cand} for {office}. {url}",
            "Your neighbors are ready to VOTE EARLY for {cand}. ARE YOU? Early voting starts {date}. {url}",
        ],
        "fundraising": [
            "{cand} here. We're {pct}% short of our end-of-month goal. Chip in ${amt} before midnight: {url}",
            "{cand} ({p1}): {a}%\n{opp} ({p2}): {b}%\n\nHelp {cand} close the gap by rushing a donation TODAY: {url}\nStop2End",
            "FINAL HOURS: we need ${amt}k to reach our goal and win in November. Rush a donation: {url}",
            "Please don't ignore this text. We can't miss our ${amt}K goal. Donate before tonight's midnight deadline >> {url}\nEnd2End",
            "It's {cand}. You haven't chipped in yet and we're TIED with {opp}. Will you donate ${amt} right now? {url}",
            "{opp} has outspent us 10 to 1. If beating them is worth even $1 to you, chip in: {url}",
        ],
        "advocacy": [
            "{org}: Tell your legislators to protect {issue}. Call {rep}'s office today at {phone}. Reply STOP to opt out.",
            "The {org} needs you. {issue} is on the line in the state legislature this week. Take action: {url}",
            "Thanks for joining the {org}! Stay tuned for ways to fight for {issue}. Reply STOP to unsubscribe.",
            "{opp} voted against {issue}. Hold them accountable. Add your name: {url}",
        ],
        "election_information": [
            "Early voting starts {date}! Find your polling place and make your plan to vote: {url}",
            "Election Day is {date}. Polls are open 7am-7pm. Not sure where to vote? {url}",
            "Hi {name}, this is {vol}, a volunteer with {org} reminding neighbors about voting. Have you made a plan to vote? Reply YES, NO, or ALREADY VOTED.",
            "Reminder: the deadline to register to vote is {date}. Check your registration: {url}",
        ],
        "polling_or_survey": [
            "{name}, who do you plan to vote for in the {year} race for {office}? A. {c1} ({p1}) B. {c2} ({p2}) C. Other/Undecided",
            "Quick survey from {pollster}: do you approve of {cand}'s job performance? Reply 1=Approve 2=Disapprove 3=Not sure. Reply STOP to end.",
            "{pollster} is asking likely voters about the race for {office}. Which candidate do you support, {c1} or {c2}? {url}",
            "Hi {name}, we're texting about the {year} election. Do you plan to vote early? Reply 1=Yes 2=No 3=Unsure",
        ],
        "petition": [
            "Sign the petition to protect {issue} before the legislature votes. Add your name: {url}",
            "{org}: Help us reach {n}0,000 signatures to demand action on {issue}. Sign now: {url}",
        ],
        "political_event": [
            "Join {cand} for a rally Saturday at {venue}. RSVP: {url}",
            "The {org} is hosting a town hall on {issue} at {venue} on {date}. Come ask your questions: {url}",
            "Volunteer with {cand}'s campaign this weekend! Free pizza and training at {venue}. Sign up: {url}",
        ],
    }[subtype]
    return g.fill(g.r.choice(t), cand=cand, p1=party, p2=g.r.choice("DRI"))


def _text_for(g: _Gen, kind: str) -> tuple[str, int | None, str, bool, str]:
    """Return (text, label, subtype, review_required, review_reason)."""
    if kind in _POSITIVES:
        return _positive_texts(g, kind), 1, kind, False, ""
    if kind == "review_contentless":
        return g.r.choice(["￼", "�", "￼￼", g.phone().lstrip("+")]), None, "", True, (
            "no_textual_content: attachment placeholder, replacement character, or bare phone number")
    if kind == "review_ambiguous":
        txt = g.fill("We are hosting a focus group for residents that pays {amt}0 dollars. Interested? Fill out a qualifying survey: {url}")
        return txt, None, "", True, "survey_topic_unclear: focus group; political subject not stated"
    pools = {
        "political_talk": (_POLITICAL_TALK, "personal"), "news_alert": (_NEWS, "other"),
        "transactional": (_TRANSACTIONAL, "transactional"), "commercial": (_COMMERCIAL, "commercial"),
        "nonpolitical_spam": (_SPAM, "nonpolitical_spam"), "other": (_OTHER, "other"),
    }
    if kind in pools:
        pool, sub = pools[kind]
        return g.fill(g.r.choice(pool)), 0, sub, False, ""
    # personal chat: mix of one-liners, two-part messages and templated messages
    roll = g.r.random()
    if roll < 0.25:
        text = g.r.choice(_SHORT_PERSONAL)
    elif roll < 0.55:
        text = f"{g.r.choice(_SHORT_PERSONAL)} {g.fill(g.r.choice(_PERSONAL_TEMPLATES))}"
    else:
        text = g.fill(g.r.choice(_PERSONAL_TEMPLATES))
        if g.r.random() < 0.4:
            text += " " + g.r.choice(_SHORT_PERSONAL)
    return text, 0, "personal", False, ""


def _scaled_counts(n: int, positive_rate: float) -> dict[str, int]:
    n_pos = round(n * positive_rate)
    pos_total = sum(_BASE_COUNTS[k] for k in _POSITIVES)
    counts = {k: max(1, round(_BASE_COUNTS[k] * n_pos / pos_total)) for k in _POSITIVES}
    counts[_POSITIVES[0]] += n_pos - sum(counts.values())  # exact positive total
    neg_scale = (n - n_pos) / (1000 - pos_total)
    for k, v in _BASE_COUNTS.items():
        if k not in _POSITIVES:
            counts[k] = round(v * neg_scale * 0.95)  # leave a remainder for personal chat
    counts["personal"] = n - sum(counts.values())
    return counts


def generate_synthetic(n: int = 1000, positive_rate: float = 0.10,
                       seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(raw, annotations)`` with ``n`` invented messages (deterministic for ``seed``).

    ``raw`` follows the raw-export schema; ``annotations`` follows the solicitation-label
    schema (``label_source = "synthetic"``). Exactly ``round(n * positive_rate)`` rows are
    positives.
    """
    g = _Gen(seed)
    counts = _scaled_counts(n, positive_rate)
    kinds = [k for k, c in counts.items() for _ in range(c)]
    g.r.shuffle(kinds)

    contacts = [(g.phone(), g.name()) for _ in range(24)]
    groups = [(f"chat{g.r.randint(10**17, 10**18 - 1)}", name) for name in
              ["Fantasy Football", "Family Group", "Weekend Plans", "Book Club"]]
    short_codes = [str(g.r.randint(20000, 99999)) for _ in range(12)]
    bulk_phones = [g.phone() for _ in range(20)]
    biz_emails = [f"alerts{i}@mail-{g.r.choice(_BRANDS).lower().split()[0]}.example" for i in range(6)]

    start = pd.Timestamp("2026-01-01", tz="America/Chicago")
    seconds = sorted(g.r.sample(range(0, 180 * 86400), n))
    rows, ann = [], []
    chat_ids: dict[str, int] = {}
    for i, (kind, sec) in enumerate(zip(kinds, seconds, strict=True), start=1):
        text, label, subtype, review, reason = _text_for(g, kind)
        if kind in ("personal", "political_talk", "news_alert"):
            if kind == "personal" and g.r.random() < 0.18:
                ident, cname = g.r.choice(groups)
                sender = g.r.choice(contacts)[0]
            else:
                sender, cname = g.r.choice(contacts)
                ident = sender
        elif kind in ("transactional", "commercial"):
            sender = g.r.choice(short_codes + biz_emails)
            ident, cname = sender, ""
        elif kind in _POSITIVES or kind == "review_ambiguous":
            sender = g.r.choice(bulk_phones + short_codes)
            ident, cname = sender, ""
        else:
            sender = g.r.choice(bulk_phones + biz_emails)
            ident, cname = sender, ""
        chat_id = chat_ids.setdefault(ident, len(chat_ids) + 1)
        rows.append({
            "message_id": i, "text": text, "is_from_me": 0, "group_title": None,
            "date": start + pd.Timedelta(seconds=sec), "sender": sender,
            "chat_id": float(chat_id), "chat_identifier": ident,
            "chat_name": cname if ident.startswith("chat") else "",
        })
        ann.append({"message_id": i, "label": label if label is not None else "", "subtype": subtype,
                    "review_required": review, "review_reason": reason,
                    "label_source": SYNTHETIC_LABEL_SOURCE})
    raw = pd.DataFrame(rows)
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_convert("America/Chicago")
    return raw, pd.DataFrame(ann)
