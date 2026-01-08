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

export default function PatientDashboard() {
  return (
    <>
    <div className="dashboard">
      {/* X axis */}
      <div className="main-layout">
        {/* 1/3 */}
        <div className="card patient-card">
          <h2>Patient Details</h2>
          <p>Name: Dewi Rahmawati</p>
          <p>Age: 45 years</p>
          <p>Height / Weight: 172 cm / 68 kg</p>

          <img
            src="https://dummyimage.com/200x300/0f172a/38bdf8&text=Human+Body"
            alt="body"
          />
          <button className="patient-button">MRI</button>
          <button className="patient-button">View History</button>
        </div>

        {/* 2/3 Y axis */}
        <div className="vital-column">
          <VitalCard title="Heart Rate" value={72} unit="bpm" />
          <VitalCard title="Oxygen Saturation" value={99} unit="%" />
          <VitalCard title="Respiratory Rate" value={18} unit="rpm" />
        </div>
      </div>
    </div>
    </>
  );
}
