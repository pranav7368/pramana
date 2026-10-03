"""Small parallel corpus for the demo service.

Deliberately carries the properties the pipeline is built to catch: numeric
conditions, an explicit negation, an identifier for exact-match retrieval, and a
gap (nothing about WiFi or office facilities) so abstention can be demonstrated
rather than described.

This is *not* the evaluation corpus — that is specified in
``docs/05_DATASET_SPEC.md``. The Hindi and Tamil text here has not been through
the translation QC pipeline and must not be used for measurement.
"""

DEMO_CORPUS: dict[str, str] = {
    "en": (
        "Claims are rejected if submitted more than 30 days after the discharge date. "
        "Rejected claims may be appealed within 60 days of the rejection notice. "
        "An appeal requires the original claim number and the rejection letter.\n\n"
        "Maternity benefits are not available during the first policy year. "
        "Policy PX-4471 covers outpatient dental treatment up to Rs. 25,000 per year.\n\n"
        "Employees may carry forward up to 15 unused leave days into the next year. "
        "Reimbursement requests must include original receipts and are settled "
        "within 21 working days."
    ),
    "hi": (
        "डिस्चार्ज की तारीख से 30 दिन बाद जमा किए गए क्लेम रिजेक्ट कर दिए जाते हैं। "
        "रिजेक्ट क्लेम की अपील रिजेक्शन नोटिस के 60 दिन के भीतर की जा सकती है। "
        "अपील के लिए मूल क्लेम नंबर और रिजेक्शन पत्र आवश्यक है।\n\n"
        "पहले पॉलिसी वर्ष में मातृत्व लाभ उपलब्ध नहीं है। "
        "पॉलिसी PX-4471 में ओपीडी दांतों का इलाज प्रति वर्ष 25,000 रुपए तक कवर है।\n\n"
        "कर्मचारी 15 दिन तक की बची हुई छुट्टी अगले वर्ष में ले जा सकते हैं। "
        "रीइंबर्समेंट के लिए मूल रसीदें आवश्यक हैं और भुगतान 21 कार्य दिवसों में होता है।"
    ),
    "ta": (
        "வெளியேற்றப்பட்ட தேதியிலிருந்து 30 நாட்களுக்குப் பிறகு சமர்ப்பிக்கப்பட்ட "
        "உரிமைகோரல்கள் நிராகரிக்கப்படும். "
        "நிராகரிக்கப்பட்ட உரிமைகோரல்களை நிராகரிப்பு அறிவிப்பிலிருந்து 60 நாட்களுக்குள் "
        "மேல்முறையீடு செய்யலாம். "
        "மேல்முறையீட்டிற்கு அசல் உரிமைகோரல் எண் மற்றும் நிராகரிப்புக் கடிதம் தேவை.\n\n"
        "முதல் பாலிசி ஆண்டில் மகப்பேறு நலன்கள் கிடைக்காது. "
        "PX-4471 பாலிசி வெளிநோயாளர் பல் சிகிச்சைக்கு ஆண்டுக்கு 25,000 ரூபாய் வரை வழங்குகிறது.\n\n"
        "ஊழியர்கள் பயன்படுத்தாத 15 நாட்கள் வரை விடுப்பை அடுத்த ஆண்டுக்கு எடுத்துச் செல்லலாம். "
        "திருப்பிச் செலுத்தும் கோரிக்கைகளுக்கு அசல் ரசீதுகள் தேவை; அவை 21 வேலை நாட்களுக்குள் "
        "தீர்க்கப்படும்."
    ),
}

DEMO_QUERIES: dict[str, list[str]] = {
    "en": [
        "When are claims rejected after discharge?",
        "How long do I have to appeal a rejected claim?",
        "Is maternity covered in the first year?",
        "What does policy PX-4471 cover?",
        "What is the WiFi password in the Chennai office?",
    ],
    "hi": [
        "क्लेम कब रिजेक्ट होते हैं?",
        "रिजेक्ट क्लेम की अपील कितने दिन में कर सकते हैं?",
        "पहले साल में मातृत्व लाभ मिलता है क्या?",
    ],
    "ta": [
        "உரிமைகோரல்கள் எப்போது நிராகரிக்கப்படும்?",
        "மேல்முறையீடு எத்தனை நாட்களுக்குள் செய்யலாம்?",
    ],
}
"""Example queries for the demo. The last English one has no answer in the corpus,
so it exercises the abstention path — the behaviour that is hardest to believe
without seeing it."""
