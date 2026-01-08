import { LineChart, Line, ResponsiveContainer } from "recharts";
import "./dashboard.css";

import UserIco from '../assets/User Icon.svg?react';
import AgeIco from '../assets/Fast Running Icon.svg?react';
import MeasureIco from '../assets/Measure Height Icon.svg?react';
import HeartRateIco from '../assets/Heart Pulse Fill.svg?react';
import HeartPlusIco from '../assets/Heart Plus Outline Icon.svg?react';
import LungIco from '../assets/Lung Icon.svg?react';

const mockWave = Array.from({ length: 30 }, (_, i) => ({
  x: i,
  y: Math.sin(i / 3) * 10 + 50,
}));

const VitalCard = ({ title, value, unit, Ico}) => (
  <div className="card vital-card">
    <div className="vital-header">
      <span className="vital-title"><Ico className="svg-icon" style={{ width:'15px', height:'15px', fill:'gray' }}/>{title}</span>
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
          <p><UserIco className="svg-icon" style={{ width:'15px', height: '15px'}}/>Name: Dewi Rahmawati</p>
          <p><AgeIco className="svg-icon" style={{ width:'15px', height: '15px'}}/>Age: 45 years</p>
          <p><MeasureIco className="svg-icon" style={{ width:'15px', height: '15px'}}/>Height / Weight: 172 cm / 68 kg</p>

          <img
            src="src/assets/growing-xray-human-body.png"
            alt="body"
          />
        </div>

        {/* Right: Vital Cards */}
        <div className="vital-column">
          <VitalCard title="Heart Rate" value={72} unit="bpm" Ico={HeartRateIco}/>
          <VitalCard title="Oxygen Saturation" value={99} unit="%" Ico={HeartPlusIco}/>
          <VitalCard title="Respiratory Rate" value={18} unit="rpm" Ico={LungIco}/>
        </div>
      </div>

      {/* BOTTOM SECTION */}
      <div className="info-row">
        <InfoBox title="History" />
        <InfoBox title="Blood Pressure" value="120 / 80 mmHg" />
        <InfoBox title="Overall SOFA Score" value="5" />
        <InfoBox title="Similar Patient" />
        <InfoBox title="AI Recommendation" value="Personalized" />
        <InfoBox title="Clinician Recommendation" />
      </div>
    </div>
  );
}
