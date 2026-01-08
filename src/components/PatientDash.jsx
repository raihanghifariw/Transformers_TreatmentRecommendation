import { LineChart, Line, ResponsiveContainer } from "recharts";
import "./dashboard.css";

const mockWave = Array.from({ length: 30 }, (_, i) => ({
  x: i,
  y: Math.sin(i / 3) * 10 + 50,
}));

const VitalCard = ({ title, value, unit }) => (
  <div className="card vital-card">
    <div className="vital-header">
      <span className="vital-title">{title}</span>
      <span className="vital-value">
        {value} <small>{unit}</small>
      </span>
    </div>

    <div className="vital-chart">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={mockWave}>
          <Line dataKey="y" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  </div>
);

const InfoBox = ({ title, value }) => (
  <div className="card info-box">
    <h4>{title}</h4>
    {value && <p>{value}</p>}
  </div>
);

export default function PatientDashboard() {
  return (
    <div className="dashboard">
      {/* TOP SECTION */}
      <div className="main-layout">
        {/* Left: Personal Detail */}
        <div className="card patient-card">
          <h2>Patient Details</h2>
          <p>Name: Dewi Rahmawati</p>
          <p>Age: 45 years</p>
          <p>Height / Weight: 172 cm / 68 kg</p>

          <img
            src="https://dummyimage.com/200x300/0f172a/38bdf8&text=Human+Body"
            alt="body"
          />
        </div>

        {/* Right: Vital Cards */}
        <div className="vital-column">
          <VitalCard title="Heart Rate" value={72} unit="bpm" />
          <VitalCard title="Oxygen Saturation" value={99} unit="%" />
          <VitalCard title="Respiratory Rate" value={18} unit="rpm" />
        </div>
      </div>

      {/* BOTTOM SECTION */}
      <div className="info-row">
        <InfoBox title="History" />
        <InfoBox title="Blood Pressure" value="120 / 80 mmHg" />
        <InfoBox title="Overall SOFA Score" value="5" />
        <InfoBox title="AI Recommendation" value="Personalized" />
        <InfoBox title="Clinician Recommendation" />
      </div>
    </div>
  );
}
