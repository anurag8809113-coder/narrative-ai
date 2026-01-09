from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def generate_pdf(filename, prediction, confidence, rationale, rows):
    c = canvas.Canvas(filename, pagesize=A4)
    w, h = A4
    y = h - 40

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Narrative Consistency Report")
    y -= 40

    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Prediction: {prediction}")
    y -= 20
    c.drawString(40, y, f"Confidence: {confidence}%")
    y -= 20
    c.drawString(40, y, "Rationale:")
    y -= 20

    text = c.beginText(60, y)
    for line in rationale.split("."):
        text.textLine(line.strip())
    c.drawText(text)
    y -= 80

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Claim-wise Results")
    y -= 20

    c.setFont("Helvetica", 10)
    for r in rows:
        if y < 80:
            c.showPage()
            y = h - 40
        c.drawString(40, y, f"- {r['Claim']}")
        y -= 12
        c.drawString(60, y, f"Label: {r['Label']}")
        y -= 12
        c.drawString(60, y, f"Reason: {r['Reason'][:80]}...")
        y -= 20

    c.save()

