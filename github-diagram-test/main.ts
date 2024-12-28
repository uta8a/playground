// ...existing code...

type hour = number;

function generateGeoJSON(t: hour) {
    const distance = 60 * t; // distance in km
    const startX = 139.7673068;
    const startY = 35.6809591;
    const targetX = 135.5023;
    const targetY = 34.6937;

    // Calculate the direction vector
    const directionX = targetX - startX;
    const directionY = targetY - startY;
    const length = Math.sqrt(directionX ** 2 + directionY ** 2);

    // Normalize the direction vector
    const unitX = directionX / length;
    const unitY = directionY / length;

    // Calculate the end coordinates
    const endX = startX + (unitX * distance / 111); // 1 degree longitude ~ 111 km
    const endY = startY + (unitY * distance / 111); // 1 degree latitude ~ 111 km

    const geojson = {
        type: "FeatureCollection",
        features: [
            {
                type: "Feature",
                id: 1,
                geometry: {
                    type: "LineString",
                    coordinates: [
                        [startX, startY],
                        [endX, endY],
                    ],
                },
            },
        ],
    };

    return geojson;
}

// Example usage
const t = 6; // Example input
console.log(JSON.stringify(generateGeoJSON(t), null, 2));

// ...existing code...
