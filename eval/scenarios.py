def u(text):
    return {"role": "user", "content": text}


def a(text):
    return {"role": "assistant", "content": text}


# Each scenario: sessions to ingest in order, then questions asked afterwards.
# expect="ABSTAIN" means the correct behaviour is admitting there's no such memory.
SCENARIOS = [
    {
        "name": "single_hop_basics",
        "sessions": [[
            u("Hi! I'm Meera and I work as a pediatric nurse in Chennai."),
            a("Nice to meet you, Meera! That must be rewarding work."),
            u("I have a golden retriever named Bruno."),
            a("Bruno sounds lovely!"),
        ]],
        "questions": [
            {"q": "What's my dog's name?", "expect": "Bruno", "category": "single-hop"},
            {"q": "What do I do for work?", "expect": "pediatric nurse", "category": "single-hop"},
        ],
    },
    {
        "name": "knowledge_update_city",
        "sessions": [
            [u("I live in Delhi right now."), a("Good to know!")],
            [u("Update — I moved to Bangalore last week!"), a("Congrats on the move!")],
        ],
        "questions": [
            {"q": "Where do I live?", "expect": "Bangalore", "category": "update"},
            {"q": "Where did I live before?", "expect": "Delhi", "category": "temporal"},
        ],
    },
    {
        "name": "knowledge_update_job",
        "sessions": [
            [u("I've been working as a barista at Blue Tokai."), a("Nice!")],
            [u("Big news: I quit the cafe and joined Zomato as a delivery ops manager."),
             a("Congratulations on the new role!")],
        ],
        "questions": [
            {"q": "Where do I work these days?", "expect": "Zomato, as a delivery ops manager", "category": "update"},
            {"q": "What was my old job?", "expect": "barista at Blue Tokai", "category": "temporal"},
        ],
    },
    {
        "name": "multi_hop_trip",
        "sessions": [
            [u("I'm planning a trek to Kedarkantha in December."), a("Exciting!")],
            [u("I bought new Quechua boots yesterday to prepare for my trek."), a("Great choice!")],
        ],
        "questions": [
            {"q": "What gear have I bought for my December trip?",
             "expect": "Quechua boots, bought to prepare for the Kedarkantha trek",
             "category": "multi-hop"},
        ],
    },
    {
        "name": "multi_hop_friend",
        "sessions": [
            [u("My best friend Zubin just got a job as a pilot with Indigo."), a("That's great news!")],
            [u("Zubin and I are planning to celebrate his new job in Goa."), a("Sounds fun!")],
        ],
        "questions": [
            {"q": "Why are we going to celebrate in Goa?",
             "expect": "to celebrate Zubin's new job as a pilot", "category": "multi-hop"},
            {"q": "What does Zubin do?", "expect": "pilot with Indigo", "category": "exact-token"},
        ],
    },
    {
        "name": "temporal_dates",
        "sessions": [
            [u("I adopted my cat Mochi two months ago."), a("Aww, welcome Mochi!")],
        ],
        "questions": [
            {"q": "Roughly when did I adopt Mochi?",
             "expect": "around June 2026 (two months before the conversation)",
             "category": "temporal"},
        ],
    },
    {
        "name": "abstention",
        "sessions": [
            [u("I love playing badminton on weekends."), a("Nice hobby!")],
        ],
        "questions": [
            {"q": "What's my favorite movie?", "expect": "ABSTAIN", "category": "abstention"},
            {"q": "What car do I drive?", "expect": "ABSTAIN", "category": "abstention"},
        ],
    },
    {
        "name": "preference_details",
        "sessions": [
            [u("I'm vegetarian, and I'm allergic to peanuts — it's serious."), a("Noted, that's important!")],
        ],
        "questions": [
            {"q": "Can you suggest a snack for me?",
             "expect": "a vegetarian, peanut-free suggestion that respects the allergy",
             "category": "single-hop"},
        ],
    },
    {
        # counting with a planned-vs-attended trap: the aggregation path must
        # count the two attended concerts and exclude the one merely booked
        "name": "counting_events",
        "sessions": [
            [u("Went to the Prateek Kuhad concert last night — incredible show!"),
             a("Sounds amazing!")],
            [u("Caught another concert yesterday, a small indie band at Fandom. Loved it."),
             a("Two great shows!")],
            [u("I just booked tickets for the Coldplay concert in January. Can't wait!"),
             a("That'll be fantastic!")],
        ],
        "questions": [
            {"q": "How many concerts have I been to?", "expect": "2",
             "category": "aggregation"},
        ],
    },
]
