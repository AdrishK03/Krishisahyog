import { initializeApp, getApps, type FirebaseApp } from "firebase/app";
import { getDatabase, ref, onValue, type Unsubscribe } from "firebase/database";

const firebaseConfig = {
  apiKey:
    (import.meta.env.VITE_FIREBASE_API_KEY as string | undefined) ??
    (import.meta.env.FIREBASE_API_KEY as string | undefined),
  databaseURL:
    (import.meta.env.VITE_FIREBASE_DATABASE_URL as string | undefined) ??
    (import.meta.env.FIREBASE_DATABASE_URL as string | undefined),
  projectId:
    (import.meta.env.VITE_FIREBASE_PROJECT_ID as string | undefined) ??
    (import.meta.env.FIREBASE_PROJECT_ID as string | undefined),
};

export function isFirebaseSoilConfigured(): boolean {
  return Boolean(firebaseConfig.apiKey && firebaseConfig.databaseURL);
}

function getApp(): FirebaseApp {
  if (!isFirebaseSoilConfigured()) {
    throw new Error("Firebase is not configured (missing VITE_FIREBASE_API_KEY or VITE_FIREBASE_DATABASE_URL)");
  }
  const existing = getApps()[0];
  if (existing) return existing;
  return initializeApp({
    apiKey: firebaseConfig.apiKey!,
    databaseURL: firebaseConfig.databaseURL!,
    projectId: firebaseConfig.projectId,
  });
}

export type PlantSensorState = {
  temperature: number | null;
  humidity: number | null;
  moisture: number | null;
  nitrogen: number | null;
  phosphorus: number | null;
  potassium: number | null;
  ph: number | null;
};

export const emptySensorState = (): PlantSensorState => ({
  temperature: null,
  humidity: null,
  moisture: null,
  nitrogen: null,
  phosphorus: null,
  potassium: null,
  ph: null,
});

function toNumber(v: unknown): number | null {
  if (v == null) return null;
  const n = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : null;
}

/** Raw ADC-style counts are often large; ppm-style values stay as-is. */
function npkFromFirebase(v: unknown): number | null {
  const n = toNumber(v);
  if (n == null) return null;
  if (Math.abs(n) > 500) return Math.round((n / 1000) * 100) / 100;
  return Math.round(n * 100) / 100;
}

export function parsePlantSnapshot(data: Record<string, unknown> | null): PlantSensorState {
  if (!data) return emptySensorState();
  const nRaw = data.npk_n ?? data.n;
  const pRaw = data.npk_p ?? data.p;
  const kRaw = data.npk_k ?? data.k;
  return {
    temperature: toNumber(data.temp ?? data.Temperature),
    humidity: toNumber(data.humidity ?? data.hum),
    moisture: toNumber(data.moisture ?? data.soil_pct),
    nitrogen: npkFromFirebase(nRaw),
    phosphorus: npkFromFirebase(pRaw),
    potassium: npkFromFirebase(kRaw),
    ph: toNumber(data.ph ?? data.soil_ph ?? data.pH),
  };
}

export type FirebaseSoilStatus = "connecting" | "live" | "no_data" | "error" | "unconfigured";

export function subscribePlantSensor(
  onData: (state: PlantSensorState) => void,
  onStatus: (status: FirebaseSoilStatus, message?: string) => void
): Unsubscribe {
  if (!isFirebaseSoilConfigured()) {
    onStatus("unconfigured", "Add VITE_FIREBASE_API_KEY and VITE_FIREBASE_DATABASE_URL to use live sensors.");
    return () => {};
  }

  onStatus("connecting");
  const app = getApp();
  const db = getDatabase(app);
  const plantRef = ref(db, "plant");

  return onValue(
    plantRef,
    (snap) => {
      const raw = snap.val() as Record<string, unknown> | null;
      if (!raw) {
        onStatus("no_data", "No data at plant/");
        onData(emptySensorState());
        return;
      }
      onData(parsePlantSnapshot(raw));
      onStatus("live", "Live — sensor synced");
    },
    (err) => {
      onStatus("error", err.message);
      onData(emptySensorState());
    }
  );
}
