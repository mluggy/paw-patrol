#!/usr/bin/env node

import axios from "axios";
import fs from "fs-extra";
import path from "path";
import crypto from "crypto";
import sharp from "sharp";
import * as tf from "@tensorflow/tfjs-node";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import { fileURLToPath } from "url";
import { input } from "@inquirer/prompts";
import { number } from "@inquirer/prompts";
import { password } from "@inquirer/prompts";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CONFIG_FILE = path.join(__dirname, "../.config.json");
const CACHE_DIR = path.join(__dirname, "../cache");
const OUTPUT_DIR = path.join(__dirname, "../output");

// Ensure cache and output directories exist
fs.ensureDirSync(CACHE_DIR);
fs.ensureDirSync(OUTPUT_DIR);

// Load configuration
let config = {};
try {
  if (fs.existsSync(CONFIG_FILE)) {
    config = JSON.parse(fs.readFileSync(CONFIG_FILE, "utf8"));
  }
} catch (error) {
  console.error("Error loading config file:", error.message);
}

// Helper function to save configuration
const saveConfig = () => {
  fs.writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2));
};

// Generate a hash for coordinates and heading
const generateHash = (lat, lng, heading) => {
  return crypto
    .createHash("md5")
    .update(`${lat}_${lng}_${heading}`)
    .digest("hex");
};

// Function to download Street View image from Google Street View API
const downloadStreetViewImage = async (lat, lng, heading, apiKey) => {
  const hash = generateHash(lat, lng, heading);
  const cachePath = path.join(CACHE_DIR, `${hash}.jpg`);

  // Check if image is already cached
  if (fs.existsSync(cachePath)) {
    return cachePath;
  }

  const url = `https://maps.googleapis.com/maps/api/streetview?size=640x640&location=${lat},${lng}&heading=${heading}&pitch=15&return_error_code=true&key=${apiKey}`;

  try {
    const response = await axios({
      method: "get",
      url,
      responseType: "arraybuffer",
      validateStatus: false, // Don't throw for error status codes
    });

    // Check for error status codes
    if (response.status !== 200) {
      if (response.status === 404) {
        // No console output for missing images
      } else {
        console.error(
          `Error downloading Street View image: HTTP ${response.status}`
        );
      }
      return null;
    }

    fs.writeFileSync(cachePath, response.data);
    return cachePath;
  } catch (error) {
    console.error(`Error downloading Street View image: ${error.message}`);
    return null;
  }
};

// Function to geocode an address
const geocodeAddress = async (address, apiKey) => {
  try {
    const url = `https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(
      address
    )}&key=${apiKey}`;
    const response = await axios.get(url);

    if (response.data.status === "OK" && response.data.results.length > 0) {
      const { lat, lng } = response.data.results[0].geometry.location;
      return { lat, lng };
    } else {
      throw new Error(`Geocoding failed: ${response.data.status}`);
    }
  } catch (error) {
    console.error(`Error geocoding address: ${error.message}`);
    throw error;
  }
};

// Function to generate coordinates in a radius
const generateCoordinatesInRadius = (
  centerLat,
  centerLng,
  radiusKm,
  numPoints = 10
) => {
  const coordinates = [];

  // Earth's radius in kilometers
  const earthRadius = 6371;

  // Convert radius from kilometers to radians
  const radiusRadians = radiusKm / earthRadius;

  // Generate points in concentric circles
  for (let r = 0; r <= radiusKm; r += radiusKm / numPoints) {
    const radiusRatio = r / radiusKm;
    const circlePoints = Math.max(6, Math.floor(12 * radiusRatio) * 2);

    for (let i = 0; i < circlePoints; i++) {
      const angle = (i * 2 * Math.PI) / circlePoints;
      const currentRadiusRadians = (radiusRatio * radiusKm) / earthRadius;

      // Calculate new position
      const lat =
        (Math.asin(
          Math.sin((centerLat * Math.PI) / 180) *
            Math.cos(currentRadiusRadians) +
            Math.cos((centerLat * Math.PI) / 180) *
              Math.sin(currentRadiusRadians) *
              Math.cos(angle)
        ) *
          180) /
        Math.PI;

      const lng =
        centerLng +
        (Math.atan2(
          Math.sin(angle) *
            Math.sin(currentRadiusRadians) *
            Math.cos((centerLat * Math.PI) / 180),
          Math.cos(currentRadiusRadians) -
            Math.sin((centerLat * Math.PI) / 180) *
              Math.sin((lat * Math.PI) / 180)
        ) *
          180) /
          Math.PI;

      coordinates.push({ lat, lng });
    }
  }

  // Add center point
  coordinates.push({ lat: centerLat, lng: centerLng });

  return coordinates;
};

// Process an image with the YOLO model
const processImage = async (imagePath, model, minConfidence) => {
  try {
    // Load and prepare the image
    const imageBuffer = fs.readFileSync(imagePath);
    const image = await tf.node.decodeImage(imageBuffer);

    // Run detection
    const predictions = await model.detect(image);

    // Filter for objects with confidence >= minConfidence
    const filteredPredictions = predictions.filter(
      (prediction) => prediction.score >= minConfidence
    );

    // Clean up
    tf.dispose(image);

    return filteredPredictions;
  } catch (error) {
    console.error(`Error processing image: ${error.message}`);
    return [];
  }
};

// Draw bounding boxes on an image
const drawBoundingBoxes = async (imagePath, predictions, outputPath) => {
  try {
    // Load image
    const image = sharp(imagePath);
    const metadata = await image.metadata();

    // Create SVG with bounding boxes
    const svgBuffer = Buffer.from(`
      <svg width="${Math.max(110, metadata.width)}" height="${metadata.height}">
        ${predictions
          .map((prediction) => {
            const [x, y, width, height] = prediction.bbox;
            const color = "green";

            // Check if the text would be off-screen (y < 16)
            const textAtBottom = y < 16;
            const textY = textAtBottom ? y + height + 12 : y - 4; // Position for text
            const rectY = textAtBottom ? y + height : y - 16; // Position for background rectangle

            return `
            <rect 
              x="${x}" y="${y}" width="${width}" height="${height}"
              stroke="${color}" 
              stroke-width="1" 
              fill="none" />
            <rect 
              x="${x}" y="${rectY}" 
              width="${Math.max(110, width)}" height="16" 
              fill="${color}" />
            <text 
              x="${x + 4}" y="${textY}" 
              font-family="Arial, Helvetica, sans-serif" 
              font-size="10px"
              font-weight="bold"
              text-rendering="geometricPrecision"
              fill="white">
              ${prediction.class} ${parseInt(prediction.score * 100)}%
            </text>
          `;
          })
          .join("")}
      </svg>
    `);

    // Composite the SVG onto the image
    await image
      .composite([{ input: svgBuffer, top: 0, left: 0 }])
      .toFile(outputPath);
  } catch (error) {
    console.error(`Error drawing bounding boxes: ${error.message}`);
  }
};

// Draw bounding boxes for a specific object class
const drawBoundingBoxesByClass = async (
  imagePath,
  predictions,
  outputPath,
  objectClass
) => {
  try {
    // Filter predictions for the specific class
    const filteredPredictions = predictions.filter(
      (prediction) => prediction.class === objectClass
    );

    if (filteredPredictions.length === 0) {
      return false; // No objects of this class found
    }

    // Load image
    const image = sharp(imagePath);
    const metadata = await image.metadata();

    // Create SVG with bounding boxes for this class only
    const svgBuffer = Buffer.from(`
      <svg width="${metadata.width}" height="${metadata.height}">
        ${filteredPredictions
          .map((prediction) => {
            const [x, y, width, height] = prediction.bbox;
            const color = "green";

            // Check if the text would be off-screen (y < 16)
            const textAtBottom = y < 16;
            const textY = textAtBottom ? y + height + 12 : y - 4; // Position for text
            const rectY = textAtBottom ? y + height : y - 16; // Position for background rectangle

            return `
            <rect 
              x="${x}" y="${y}" width="${width}" height="${height}"
              stroke="${color}" 
              stroke-width="1" 
              fill="none" />
            <rect 
              x="${x}" y="${rectY}" 
              width="${Math.min(220, width)}" height="16" 
              fill="${color}" />
            <text 
              x="${x + 4}" y="${textY}" 
              font-family="Arial, Helvetica, sans-serif" 
              font-size="10px"
              font-weight="bold"
              text-rendering="geometricPrecision"
              fill="white">
              ${parseInt(prediction.score * 100)}% ${prediction.class}
            </text>
          `;
          })
          .join("")}
      </svg>
    `);

    // Composite the SVG onto the image
    await image
      .composite([{ input: svgBuffer, top: 0, left: 0 }])
      .toFile(outputPath);
    return true;
  } catch (error) {
    console.error(
      `Error drawing bounding boxes for ${objectClass}: ${error.message}`
    );
    return false;
  }
};

// Main function
const main = async () => {
  try {
    // Prompt for Google Maps API key if not exists
    if (!config.google_maps_api_key) {
      const apiKey = await password({
        message: "Enter your Google Maps API key:",
        mask: "*",
        validate: (input) =>
          input.trim() !== "" ? true : "API key is required",
      });
      config.google_maps_api_key = apiKey;
      saveConfig();
    }

    // Prompt for address
    const address = await input({
      message: "Enter an address:",
      default: config.address || "",
      validate: (input) => (input.trim() !== "" ? true : "Address is required"),
    });
    config.address = address;
    saveConfig();

    // Prompt for radius in km
    const radius = await number({
      message: "Enter radius in km:",
      default: config.radius || 1,
      validate: (input) => (input > 0 ? true : "Radius must be greater than 0"),
    });
    config.radius = radius;
    saveConfig();

    // Prompt for minimum confidence
    const confidence = await number({
      message: "Enter minimum confidence (0 to 1.00):",
      default: config.confidence || 0.5,
      float: true,
      step: 0.01,
      validate: (input) =>
        input >= 0 && input <= 1
          ? true
          : "Confidence must be between 0 and 1.00",
    });
    config.confidence = confidence;
    saveConfig();

    // Geocoding address
    const { lat, lng } = await geocodeAddress(
      address,
      config.google_maps_api_key
    );

    // Load COCO-SSD model
    const model = await cocoSsd.load();

    // Generate coordinates in radius
    const coordinates = generateCoordinatesInRadius(lat, lng, radius);

    // Headings (0, 90, 180, 270 for N, E, S, W)
    const headings = [0, 90, 180, 270];

    // Track total objects found by class
    const totalObjectCounts = {};

    // Counter for processed locations
    let processedLocations = 0;
    let totalLocations = coordinates.length * headings.length;

    // Process a single coordinate
    const processCoordinate = async (coord) => {
      // Object counts for this specific coordinate
      const coordObjectCounts = {};
      let hasProcessedAnyImageForThisCoord = false;

      for (const heading of headings) {
        const imagePath = await downloadStreetViewImage(
          coord.lat,
          coord.lng,
          heading,
          config.google_maps_api_key
        );

        if (!imagePath) {
          continue;
        }

        hasProcessedAnyImageForThisCoord = true;
        processedLocations++;

        const predictions = await processImage(imagePath, model, confidence);

        if (predictions.length > 0) {
          // Get counts by class
          const classCounts = {};
          predictions.forEach((p) => {
            classCounts[p.class] = (classCounts[p.class] || 0) + 1;
            totalObjectCounts[p.class] = (totalObjectCounts[p.class] || 0) + 1;
          });

          // Update coordinate object counts
          Object.keys(classCounts).forEach((cls) => {
            coordObjectCounts[cls] =
              (coordObjectCounts[cls] || 0) + classCounts[cls];
          });

          // Get unique object classes in this image
          const uniqueClasses = [...new Set(predictions.map((p) => p.class))];

          // Process each class separately and save to class-specific folder
          for (const objectClass of uniqueClasses) {
            const classDir = path.join(OUTPUT_DIR, objectClass);
            fs.ensureDirSync(classDir);

            const outputPath = path.join(
              classDir,
              `${coord.lat.toFixed(6)}_${coord.lng.toFixed(6)}_${heading}.jpg`
            );

            await drawBoundingBoxesByClass(
              imagePath,
              predictions,
              outputPath,
              objectClass
            );
          }
        }
      }
    };

    // Process coordinates in batches of 5 in parallel
    const BATCH_SIZE = 5;
    for (let i = 0; i < coordinates.length; i += BATCH_SIZE) {
      const batch = coordinates.slice(i, i + BATCH_SIZE);
      await Promise.all(batch.map((coord) => processCoordinate(coord)));
    }
  } catch (error) {
    console.error("An error occurred:", error.message);
  }
};

main();
