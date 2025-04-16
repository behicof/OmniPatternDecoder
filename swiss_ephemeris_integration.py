import swisseph as swe
from datetime import datetime
import numpy as np

class SwissEphemerisConnector:
    def __init__(self, ephemeris_path=None):
        """
        Initialize Swiss Ephemeris connector
        
        Parameters:
        ephemeris_path: Path to ephemeris files (optional)
        """
        if ephemeris_path:
            swe.set_ephe_path(ephemeris_path)
        
        # Planetary body constants
        self.SUN = swe.SUN
        self.MOON = swe.MOON
        self.MERCURY = swe.MERCURY
        self.VENUS = swe.VENUS
        self.MARS = swe.MARS
        self.JUPITER = swe.JUPITER
        self.SATURN = swe.SATURN
        self.URANUS = swe.URANUS
        self.NEPTUNE = swe.NEPTUNE
        self.PLUTO = swe.PLUTO
        
        # Aspect angles (in degrees)
        self.CONJUNCTION = 0
        self.OPPOSITION = 180
        self.TRINE = 120
        self.SQUARE = 90
        self.SEXTILE = 60
        
        # Aspect orbs (allowable deviation in degrees)
        self.orbs = {
            self.CONJUNCTION: 8,
            self.OPPOSITION: 8, 
            self.TRINE: 7,
            self.SQUARE: 7,
            self.SEXTILE: 6
        }
    
    def date_to_julian(self, date):
        """Convert datetime to Julian day"""
        return swe.julday(
            date.year, 
            date.month, 
            date.day, 
            date.hour + date.minute/60 + date.second/3600
        )
    
    def get_planet_position(self, planet, date, flags=swe.FLG_SWIEPH):
        """
        Get planet position in the zodiac
        
        Parameters:
        planet: Planet constant (e.g., swe.SUN)
        date: Python datetime object
        flags: Swiss Ephemeris calculation flags
        
        Returns:
        Dictionary with position data
        """
        jd = self.date_to_julian(date)
        
        # Calculate position
        result = swe.calc_ut(jd, planet, flags)
        
        # Extract longitude (position in the zodiac)
        longitude = result[0]
        
        # Determine zodiac sign
        zodiac_sign_num = int(longitude / 30)
        zodiac_signs = [
            'Aries', 'Taurus', 'Gemini', 'Cancer', 
            'Leo', 'Virgo', 'Libra', 'Scorpio', 
            'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces'
        ]
        zodiac_sign = zodiac_signs[zodiac_sign_num]
        
        # Determine if in fire sign
        fire_signs = ['Aries', 'Leo', 'Sagittarius']
        in_fire_sign = zodiac_sign in fire_signs
        
        return {
            'longitude': longitude,
            'zodiac_sign': zodiac_sign,
            'zodiac_degree': longitude % 30,
            'in_fire_sign': in_fire_sign,
            'retrograde': result[3] < 0,
            'speed': result[3]
        }
    
    def check_aspect(self, angle1, angle2, aspect_type):
        """
        Check if two planets form an aspect
        
        Parameters:
        angle1, angle2: Longitudes of the two planets
        aspect_type: Type of aspect to check (e.g., CONJUNCTION)
        
        Returns:
        True if the aspect is formed, False otherwise
        """
        # Calculate the angular difference
        diff = abs(angle1 - angle2) % 360
        if diff > 180:
            diff = 360 - diff
        
        # Check if within orb
        target_angle = aspect_type
        orb = self.orbs.get(aspect_type, 6)
        
        return abs(diff - target_angle) <= orb
    
    def get_all_planets_positions(self, date):
        """
        Get positions of all major planets
        
        Parameters:
        date: Python datetime object
        
        Returns:
        Dictionary with position data for all planets
        """
        planets = {
            'Sun': self.SUN,
            'Moon': self.MOON,
            'Mercury': self.MERCURY,
            'Venus': self.VENUS,
            'Mars': self.MARS,
            'Jupiter': self.JUPITER,
            'Saturn': self.SATURN,
            'Uranus': self.URANUS,
            'Neptune': self.NEPTUNE,
            'Pluto': self.PLUTO
        }
        
        results = {}
        for name, planet in planets.items():
            results[name] = self.get_planet_position(planet, date)
        
        return results
    
    def find_all_aspects(self, date):
        """
        Find all aspects between planets on a given date
        
        Parameters:
        date: Python datetime object
        
        Returns:
        List of aspects found
        """
        positions = self.get_all_planets_positions(date)
        aspects = []
        
        planet_names = list(positions.keys())
        aspect_names = {
            self.CONJUNCTION: 'Conjunction',
            self.OPPOSITION: 'Opposition',
            self.TRINE: 'Trine',
            self.SQUARE: 'Square',
            self.SEXTILE: 'Sextile'
        }
        
        # Check aspects between every pair of planets
        for i in range(len(planet_names)):
            for j in range(i + 1, len(planet_names)):
                planet1 = planet_names[i]
                planet2 = planet_names[j]
                
                angle1 = positions[planet1]['longitude']
                angle2 = positions[planet2]['longitude']
                
                for aspect_type, aspect_name in aspect_names.items():
                    if self.check_aspect(angle1, angle2, aspect_type):
                        aspects.append({
                            'planet1': planet1,
                            'planet2': planet2,
                            'aspect': aspect_name,
                            'orb': abs((abs(angle1 - angle2) % 360) - aspect_type)
                        })
        
        return aspects
    
    def get_data_for_date_range(self, start_date, end_date, days_step=1):
        """
        Get astronomical data for a range of dates
        
        Parameters:
        start_date: Starting datetime
        end_date: Ending datetime
        days_step: Step size in days
        
        Returns:
        Dictionary with dates and astronomical data
        """
        from datetime import timedelta
        
        results = []
        current_date = start_date
        
        while current_date <= end_date:
            # Get positions
            positions = self.get_all_planets_positions(current_date)
            
            # Get aspects
            aspects = self.find_all_aspects(current_date)
            
            # Record data
            results.append({
                'date': current_date,
                'positions': positions,
                'aspects': aspects
            })
            
            # Move to next date
            current_date += timedelta(days=days_step)
        
        return results

    def close(self):
        """Close Swiss Ephemeris connection"""
        swe.close()

# Example usage
if __name__ == "__main__":
    eph = SwissEphemerisConnector()
    
    # Get current planetary positions
    now = datetime.now()
    positions = eph.get_all_planets_positions(now)
    
    print(f"Planetary positions for {now}:")
    for planet, data in positions.items():
        print(f"{planet}: {data['longitude']:.2f}° ({data['zodiac_sign']} {data['zodiac_degree']:.2f}°)")
        if data.get('retrograde'):
            print(f"  Retrograde (speed: {data['speed']:.4f})")
    
    # Get current aspects
    aspects = eph.find_all_aspects(now)
    
    print("\nCurrent planetary aspects:")
    for aspect in aspects:
        print(f"{aspect['planet1']} {aspect['aspect']} {aspect['planet2']} (orb: {aspect['orb']:.2f}°)")
    
    eph.close()