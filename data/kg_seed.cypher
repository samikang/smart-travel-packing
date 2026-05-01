// ==========================================
// PackPal Knowledge Graph — Complete Seed
// ==========================================
// Run this in Neo4j Browser (Query tab) to seed all nodes and relationships.
// Uses MERGE so it is safe to re-run — existing nodes are updated, not duplicated.

// ── Constraints (idempotent) ──────────────────────────────────────────────────
CREATE CONSTRAINT IF NOT EXISTS FOR (g:GarmentType)  REQUIRE g.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (f:FareClass)    REQUIRE f.type IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (a:Airline)      REQUIRE a.iata IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (w:WeatherCondition) REQUIRE w.condition_type IS UNIQUE;

// ── 1. ALL 37 GarmentType nodes (ISO 9920 / ASHRAE base CLO) ─────────────────
// Existing 12 nodes are updated via MERGE + SET (safe).

// Base layer — tops
MERGE (g:GarmentType {name: "T-shirt or short sleeves"})       SET g.base_clo = 0.09, g.category = "base";
MERGE (g:GarmentType {name: "Long-sleeve shirt"})               SET g.base_clo = 0.12, g.category = "base";
MERGE (g:GarmentType {name: "Thermal underlayer"})              SET g.base_clo = 0.20, g.category = "base";
MERGE (g:GarmentType {name: "Lightweight breathable clothing"}) SET g.base_clo = 0.05, g.category = "base";
MERGE (g:GarmentType {name: "Business attire"})                 SET g.base_clo = 0.25, g.category = "base";
MERGE (g:GarmentType {name: "Casual wear"})                     SET g.base_clo = 0.15, g.category = "base";
MERGE (g:GarmentType {name: "Smart casual outfit"})             SET g.base_clo = 0.20, g.category = "base";

// Base layer — bottoms
MERGE (g:GarmentType {name: "Jeans or trousers"})               SET g.base_clo = 0.20, g.category = "base";
MERGE (g:GarmentType {name: "Shorts or light trousers"})        SET g.base_clo = 0.06, g.category = "base";

// Mid layer
MERGE (g:GarmentType {name: "Warm sweater or fleece"})          SET g.base_clo = 0.30, g.category = "mid";
MERGE (g:GarmentType {name: "Light jacket or fleece"})          SET g.base_clo = 0.25, g.category = "mid";
MERGE (g:GarmentType {name: "Windproof jacket"})                SET g.base_clo = 0.25, g.category = "mid";

// Outer layer
MERGE (g:GarmentType {name: "Heavy winter coat"})               SET g.base_clo = 0.35, g.category = "outer";
MERGE (g:GarmentType {name: "Waterproof jacket"})               SET g.base_clo = 0.20, g.category = "outer";
MERGE (g:GarmentType {name: "Smart jacket"})                    SET g.base_clo = 0.30, g.category = "outer";

// Footwear
MERGE (g:GarmentType {name: "Insulated boots"})                 SET g.base_clo = 0.40, g.category = "feet";
MERGE (g:GarmentType {name: "Comfortable walking shoes"})       SET g.base_clo = 0.04, g.category = "feet";
MERGE (g:GarmentType {name: "Waterproof snow boots"})           SET g.base_clo = 0.35, g.category = "feet";
MERGE (g:GarmentType {name: "Formal shoes"})                    SET g.base_clo = 0.04, g.category = "feet";

// Accessories
MERGE (g:GarmentType {name: "Gloves and scarf"})                SET g.base_clo = 0.10, g.category = "acc";
MERGE (g:GarmentType {name: "Thermal socks"})                   SET g.base_clo = 0.05, g.category = "feet";

// Packing items (CLO not applicable — stored as 0.0 for schema consistency)
MERGE (g:GarmentType {name: "Compact umbrella"})                SET g.base_clo = 0.0,  g.category = "gear";
MERGE (g:GarmentType {name: "Waterproof bag cover"})            SET g.base_clo = 0.0,  g.category = "gear";
MERGE (g:GarmentType {name: "Sunscreen SPF 50+"})               SET g.base_clo = 0.0,  g.category = "gear";
MERGE (g:GarmentType {name: "Sunscreen SPF 30+"})               SET g.base_clo = 0.0,  g.category = "gear";
MERGE (g:GarmentType {name: "Sunglasses"})                      SET g.base_clo = 0.0,  g.category = "gear";
MERGE (g:GarmentType {name: "Wide-brim hat"})                   SET g.base_clo = 0.0,  g.category = "gear";
MERGE (g:GarmentType {name: "Hand warmers"})                    SET g.base_clo = 0.0,  g.category = "gear";
MERGE (g:GarmentType {name: "Laptop bag"})                      SET g.base_clo = 0.0,  g.category = "gear";
MERGE (g:GarmentType {name: "Power adapter"})                   SET g.base_clo = 0.0,  g.category = "gear";
MERGE (g:GarmentType {name: "Business cards"})                  SET g.base_clo = 0.0,  g.category = "gear";
MERGE (g:GarmentType {name: "Day backpack"})                    SET g.base_clo = 0.0,  g.category = "gear";
MERGE (g:GarmentType {name: "Phone charger / power bank"})      SET g.base_clo = 0.0,  g.category = "gear";
MERGE (g:GarmentType {name: "City map or offline maps"})        SET g.base_clo = 0.0,  g.category = "gear";
MERGE (g:GarmentType {name: "Reusable water bottle"})           SET g.base_clo = 0.0,  g.category = "gear";
MERGE (g:GarmentType {name: "Small gift (optional)"})           SET g.base_clo = 0.0,  g.category = "gear";
MERGE (g:GarmentType {name: "Phone charger"})                   SET g.base_clo = 0.0,  g.category = "gear";

// ── 2. FareClass nodes ────────────────────────────────────────────────────────
MERGE (:FareClass {type: "economy"});
MERGE (:FareClass {type: "premium_economy"});
MERGE (:FareClass {type: "business"});
MERGE (:FareClass {type: "first"});

// ── 3. Airline nodes (existing 6 + 5 new) ────────────────────────────────────
MERGE (:Airline {name: "Singapore Airlines", iata: "SQ"});
MERGE (:Airline {name: "Scoot",              iata: "TR"});
MERGE (:Airline {name: "Emirates",           iata: "EK"});
MERGE (:Airline {name: "Delta Air Lines",    iata: "DL"});
MERGE (:Airline {name: "EasyJet",            iata: "U2"});
MERGE (:Airline {name: "Qatar Airways",      iata: "QR"});
MERGE (:Airline {name: "Jetstar",            iata: "3K"});
MERGE (:Airline {name: "AirAsia",            iata: "AK"});
MERGE (:Airline {name: "Cathay Pacific",     iata: "CX"});
MERGE (:Airline {name: "Qantas",             iata: "QF"});
MERGE (:Airline {name: "Lufthansa",          iata: "LH"});

// ── 4. Baggage Limits (HAS_LIMIT relationships) ───────────────────────────────
// Format: checked_kg / carry_on_kg  (per session_clarifications.txt)

// Singapore Airlines
MATCH (a:Airline {iata:"SQ"}), (f:FareClass {type:"economy"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=25, r.carry_on_kg=7;
MATCH (a:Airline {iata:"SQ"}), (f:FareClass {type:"premium_economy"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=35, r.carry_on_kg=7;
MATCH (a:Airline {iata:"SQ"}), (f:FareClass {type:"business"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=35, r.carry_on_kg=7;
MATCH (a:Airline {iata:"SQ"}), (f:FareClass {type:"first"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=50, r.carry_on_kg=7;

// Scoot
MATCH (a:Airline {iata:"TR"}), (f:FareClass {type:"economy"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=15, r.carry_on_kg=10;
MATCH (a:Airline {iata:"TR"}), (f:FareClass {type:"business"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=25, r.carry_on_kg=10;

// Jetstar
MATCH (a:Airline {iata:"3K"}), (f:FareClass {type:"economy"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=15, r.carry_on_kg=7;
MATCH (a:Airline {iata:"3K"}), (f:FareClass {type:"business"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=30, r.carry_on_kg=7;

// AirAsia
MATCH (a:Airline {iata:"AK"}), (f:FareClass {type:"economy"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=15, r.carry_on_kg=7;
MATCH (a:Airline {iata:"AK"}), (f:FareClass {type:"business"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=30, r.carry_on_kg=7;

// Cathay Pacific
MATCH (a:Airline {iata:"CX"}), (f:FareClass {type:"economy"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=23, r.carry_on_kg=7;
MATCH (a:Airline {iata:"CX"}), (f:FareClass {type:"business"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=30, r.carry_on_kg=7;
MATCH (a:Airline {iata:"CX"}), (f:FareClass {type:"first"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=40, r.carry_on_kg=7;

// Emirates
MATCH (a:Airline {iata:"EK"}), (f:FareClass {type:"economy"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=25, r.carry_on_kg=7;
MATCH (a:Airline {iata:"EK"}), (f:FareClass {type:"business"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=40, r.carry_on_kg=7;
MATCH (a:Airline {iata:"EK"}), (f:FareClass {type:"first"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=50, r.carry_on_kg=7;

// Qantas
MATCH (a:Airline {iata:"QF"}), (f:FareClass {type:"economy"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=23, r.carry_on_kg=7;
MATCH (a:Airline {iata:"QF"}), (f:FareClass {type:"business"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=40, r.carry_on_kg=7;
MATCH (a:Airline {iata:"QF"}), (f:FareClass {type:"first"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=50, r.carry_on_kg=7;

// Lufthansa
MATCH (a:Airline {iata:"LH"}), (f:FareClass {type:"economy"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=23, r.carry_on_kg=8;
MATCH (a:Airline {iata:"LH"}), (f:FareClass {type:"business"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=32, r.carry_on_kg=8;
MATCH (a:Airline {iata:"LH"}), (f:FareClass {type:"first"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=40, r.carry_on_kg=8;

// Delta Air Lines
MATCH (a:Airline {iata:"DL"}), (f:FareClass {type:"economy"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=23, r.carry_on_kg=7;
MATCH (a:Airline {iata:"DL"}), (f:FareClass {type:"business"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=32, r.carry_on_kg=7;
MATCH (a:Airline {iata:"DL"}), (f:FareClass {type:"first"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=45, r.carry_on_kg=7;

// Qatar Airways
MATCH (a:Airline {iata:"QR"}), (f:FareClass {type:"economy"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=23, r.carry_on_kg=7;
MATCH (a:Airline {iata:"QR"}), (f:FareClass {type:"business"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=40, r.carry_on_kg=7;
MATCH (a:Airline {iata:"QR"}), (f:FareClass {type:"first"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=50, r.carry_on_kg=7;

// EasyJet (economy only — budget carrier)
MATCH (a:Airline {iata:"U2"}), (f:FareClass {type:"economy"})
  MERGE (a)-[r:HAS_LIMIT]->(f) SET r.checked_kg=23, r.carry_on_kg=15;

// ── 5. WeatherCondition nodes + CLO layering rules ────────────────────────────
// Each condition links to the garments required, tagged with layer_role.

MERGE (w:WeatherCondition {condition_type: "freezing"})
  SET w.temp_min = -30, w.temp_max = 0;
MERGE (w:WeatherCondition {condition_type: "cold"})
  SET w.temp_min = 0, w.temp_max = 10;
MERGE (w:WeatherCondition {condition_type: "cool"})
  SET w.temp_min = 10, w.temp_max = 17;
MERGE (w:WeatherCondition {condition_type: "mild"})
  SET w.temp_min = 17, w.temp_max = 24;
MERGE (w:WeatherCondition {condition_type: "warm"})
  SET w.temp_min = 24, w.temp_max = 32;
MERGE (w:WeatherCondition {condition_type: "hot"})
  SET w.temp_min = 32, w.temp_max = 50;
MERGE (w:WeatherCondition {condition_type: "rainy"})
  SET w.temp_min = -10, w.temp_max = 50;
MERGE (w:WeatherCondition {condition_type: "snowy"})
  SET w.temp_min = -30, w.temp_max = 3;

// CLO layering: freezing
MATCH (w:WeatherCondition {condition_type:"freezing"}), (g:GarmentType {name:"Thermal underlayer"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "base";
MATCH (w:WeatherCondition {condition_type:"freezing"}), (g:GarmentType {name:"Warm sweater or fleece"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "mid";
MATCH (w:WeatherCondition {condition_type:"freezing"}), (g:GarmentType {name:"Heavy winter coat"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "outer";
MATCH (w:WeatherCondition {condition_type:"freezing"}), (g:GarmentType {name:"Insulated boots"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "feet";
MATCH (w:WeatherCondition {condition_type:"freezing"}), (g:GarmentType {name:"Gloves and scarf"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "acc";

// CLO layering: cold
MATCH (w:WeatherCondition {condition_type:"cold"}), (g:GarmentType {name:"Long-sleeve shirt"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "base";
MATCH (w:WeatherCondition {condition_type:"cold"}), (g:GarmentType {name:"Warm sweater or fleece"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "mid";
MATCH (w:WeatherCondition {condition_type:"cold"}), (g:GarmentType {name:"Heavy winter coat"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "outer";

// CLO layering: cool
MATCH (w:WeatherCondition {condition_type:"cool"}), (g:GarmentType {name:"Long-sleeve shirt"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "base";
MATCH (w:WeatherCondition {condition_type:"cool"}), (g:GarmentType {name:"Light jacket or fleece"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "mid";
MATCH (w:WeatherCondition {condition_type:"cool"}), (g:GarmentType {name:"Jeans or trousers"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "base";

// CLO layering: mild
MATCH (w:WeatherCondition {condition_type:"mild"}), (g:GarmentType {name:"Long-sleeve shirt"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "base";
MATCH (w:WeatherCondition {condition_type:"mild"}), (g:GarmentType {name:"Light jacket or fleece"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "outer";

// CLO layering: warm/hot
MATCH (w:WeatherCondition {condition_type:"warm"}), (g:GarmentType {name:"T-shirt or short sleeves"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "base";
MATCH (w:WeatherCondition {condition_type:"warm"}), (g:GarmentType {name:"Shorts or light trousers"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "base";
MATCH (w:WeatherCondition {condition_type:"hot"}), (g:GarmentType {name:"Lightweight breathable clothing"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "base";

// CLO layering: rainy
MATCH (w:WeatherCondition {condition_type:"rainy"}), (g:GarmentType {name:"Waterproof jacket"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "outer";
MATCH (w:WeatherCondition {condition_type:"rainy"}), (g:GarmentType {name:"Compact umbrella"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "gear";

// CLO layering: snowy
MATCH (w:WeatherCondition {condition_type:"snowy"}), (g:GarmentType {name:"Waterproof snow boots"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "feet";
MATCH (w:WeatherCondition {condition_type:"snowy"}), (g:GarmentType {name:"Thermal socks"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "feet";
MATCH (w:WeatherCondition {condition_type:"snowy"}), (g:GarmentType {name:"Heavy winter coat"})
  MERGE (w)-[r:REQUIRES]->(g) SET r.layer_role = "outer";
