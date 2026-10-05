from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "model.pkl")
preprocessor = joblib.load(BASE_DIR / "preprocessor.pkl")

app = FastAPI(title="Flight Price Prediction API")


class FlightInput(BaseModel):
    airline: str
    source: str
    departure: str
    stops: Literal["zero", "one", "two_or_more"]
    arrival: str
    destination: str
    # Accept JSON key "class", because class is a Python keyword.
    travel_class: str = Field(alias="class")
    duration: float = Field(gt=0, allow_inf_nan=False)
    days_left: int = Field(ge=0)


@app.get("/")
def home():
    return {"message": "Flight Price Prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(flight: FlightInput):
    if flight.source == flight.destination:
        raise HTTPException(
            status_code=422,
            detail="Source and destination must be different.",
        )

    # Use the original column names expected by the preprocessor.
    data = pd.DataFrame([flight.model_dump(by_alias=True)])

    try:
        processed = preprocessor.transform(data)
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail="Invalid flight features or an unsupported category.",
        ) from error

    prediction = model.predict(processed)[0]

    return {
        "predicted_price": round(float(prediction), 2),
        "currency": "INR",
    }

from fastapi.responses import HTMLResponse


@app.get("/test", response_class=HTMLResponse)
def test_page():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Flight Price Predictor</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #eef3fa;
            color: #172b4d;
        }
        main {
            max-width: 850px;
            margin: 45px auto;
            padding: 0 20px;
        }
        .header { margin-bottom: 25px; }
        .badge {
            display: inline-block;
            background: #dbeafe;
            color: #1d4ed8;
            padding: 7px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        h1 { margin: 16px 0 10px; font-size: 34px; }
        .subtitle { color: #64748b; line-height: 1.6; }
        .card {
            background: white;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 12px 35px #17314b12;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        label {
            display: block;
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 8px;
        }
        select, input {
            width: 100%;
            padding: 13px;
            background: #f8fafc;
            color: #172b4d;
            border: 1px solid #d6e0ec;
            border-radius: 10px;
            font-size: 15px;
        }
        select:focus, input:focus {
            outline: 3px solid #bfdbfe;
            border-color: #2563eb;
        }
        .hint {
            font-size: 12px;
            color: #64748b;
            margin-top: 7px;
        }
        button {
            width: 100%;
            margin-top: 25px;
            padding: 15px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
        }
        button:hover { background: #1d4ed8; }
        button:disabled { opacity: .6; cursor: wait; }
        .result {
            margin-top: 25px;
            padding: 25px;
            border-radius: 14px;
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
        }
        .result-title { color: #047857; font-size: 14px; }
        .price {
            color: #065f46;
            font-size: 38px;
            font-weight: bold;
            margin-top: 8px;
        }
        .error {
            margin-top: 20px;
            padding: 15px;
            background: #fef2f2;
            color: #b91c1c;
            border-radius: 10px;
        }
        .footer {
            text-align: center;
            color: #64748b;
            font-size: 12px;
            margin-top: 22px;
            line-height: 1.6;
        }
        [hidden] { display: none !important; }
        @media (max-width: 600px) {
            main { margin: 25px auto; }
            .grid { grid-template-columns: 1fr; }
            .card { padding: 22px; }
            h1 { font-size: 28px; }
        }
    </style>
</head>
<body>
<main>
    <header class="header">
        <span class="badge">FLIGHT PRICE PREDICTOR</span>
        <h1>Plan your next flight</h1>
        <p class="subtitle">
            Enter your journey details to estimate the ticket price.
        </p>
    </header>

    <section class="card">
        <form id="flight-form">
            <div class="grid">
                <div>
                    <label for="airline">Airline</label>
                    <select id="airline" required>
                        <option value="Vistara">Vistara</option>
                        <option value="Air_India">Air India</option>
                        <option value="Indigo">Indigo</option>
                        <option value="GO_FIRST">GO FIRST</option>
                        <option value="AirAsia">AirAsia</option>
                        <option value="SpiceJet">SpiceJet</option>
                    </select>
                </div>

                <div>
                    <label for="travel-class">Travel class</label>
                    <select id="travel-class" required>
                        <option value="Economy">Economy</option>
                        <option value="Business">Business</option>
                    </select>
                </div>

                <div>
                    <label for="source">From</label>
                    <select id="source" required>
                        <option>Bangalore</option>
                        <option>Delhi</option>
                        <option>Mumbai</option>
                        <option>Kolkata</option>
                        <option>Hyderabad</option>
                        <option>Chennai</option>
                    </select>
                </div>

                <div>
                    <label for="destination">To</label>
                    <select id="destination" required>
                        <option>Delhi</option>
                        <option>Bangalore</option>
                        <option>Mumbai</option>
                        <option>Kolkata</option>
                        <option>Hyderabad</option>
                        <option>Chennai</option>
                    </select>
                </div>

                <div>
                    <label for="departure">Departure time</label>
                    <select id="departure" required>
                        <option value="Morning">Morning</option>
                        <option value="Early_Morning">Early morning</option>
                        <option value="Afternoon">Afternoon</option>
                        <option value="Evening">Evening</option>
                        <option value="Night">Night</option>
                        <option value="Late_Night">Late night</option>
                    </select>
                </div>

                <div>
                    <label for="arrival">Arrival time</label>
                    <select id="arrival" required>
                        <option value="Afternoon">Afternoon</option>
                        <option value="Early_Morning">Early morning</option>
                        <option value="Morning">Morning</option>
                        <option value="Evening">Evening</option>
                        <option value="Night">Night</option>
                        <option value="Late_Night">Late night</option>
                    </select>
                </div>

                <div>
                    <label for="stops">Stops</label>
                    <select id="stops" required>
                        <option value="zero">Non-stop</option>
                        <option value="one">One stop</option>
                        <option value="two_or_more">Two or more stops</option>
                    </select>
                </div>

                <div>
                    <label for="duration">Duration (hours)</label>
                    <input id="duration" type="number"
                           min="0.01" step="0.01" value="2.67" required>
                    <div class="hint">2 hours 30 minutes = 2.5 hours</div>
                </div>

                <div>
                    <label for="days-left">Days until departure</label>
                    <input id="days-left" type="number"
                           min="0" step="1" value="18" required>
                </div>
            </div>

            <button id="submit-button" type="submit">
                Predict ticket price
            </button>
        </form>

        <div id="result" class="result" role="status" hidden>
            <div class="result-title">ESTIMATED TICKET PRICE</div>
            <div id="price" class="price"></div>
        </div>

        <div id="error" class="error" role="alert" hidden></div>
    </section>

    <p class="footer">
        Predictions use historical training data.<br>
        This estimate is not a live fare or booking quote.
    </p>
</main>

<script>
    const form = document.getElementById("flight-form");
    const button = document.getElementById("submit-button");
    const result = document.getElementById("result");
    const errorBox = document.getElementById("error");

    function getValue(id) {
        return document.getElementById(id).value;
    }

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        result.hidden = true;
        errorBox.hidden = true;

        const payload = {
            airline: getValue("airline"),
            source: getValue("source"),
            departure: getValue("departure"),
            stops: getValue("stops"),
            arrival: getValue("arrival"),
            destination: getValue("destination"),
            class: getValue("travel-class"),
            duration: Number(getValue("duration")),
            days_left: Number(getValue("days-left"))
        };

        if (payload.source === payload.destination) {
            errorBox.textContent = "Choose different source and destination cities.";
            errorBox.hidden = false;
            return;
        }

        button.disabled = true;
        button.textContent = "Predicting...";

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok) {
                let message = "Prediction failed. Please check your inputs.";
                if (typeof data.detail === "string") {
                    message = data.detail;
                } else if (Array.isArray(data.detail)) {
                    message = data.detail.map(item => item.msg).join("; ");
                }
                throw new Error(message);
            }

            if (typeof data.predicted_price !== "number" ||
                !Number.isFinite(data.predicted_price)) {
                throw new Error("The API returned an invalid price.");
            }

            document.getElementById("price").textContent =
                new Intl.NumberFormat("en-IN", {
                    style: "currency",
                    currency: "INR",
                    maximumFractionDigits: 2
                }).format(data.predicted_price);

            result.hidden = false;
        } catch (error) {
            errorBox.textContent = error.message;
            errorBox.hidden = false;
        } finally {
            button.disabled = false;
            button.textContent = "Predict ticket price";
        }
    });
</script>
</body>
</html>
"""