# Database initialization and migration
from sqlalchemy import create_engine, text
from .config_manager import config_manager
import logging

logger = logging.getLogger(__name__)

def initialize_database():
    """Initialize database with SDG data"""
    engine = create_engine(config_manager.get_database_url())
    
    with engine.begin() as conn:
        # Initialize SDG goals
        sdg_data = [
            (1, "No Poverty", "#E5243B", "End poverty in all its forms everywhere"),
            (2, "Zero Hunger", "#DDA63A", "End hunger, achieve food security and improved nutrition and promote sustainable agriculture"),
            (3, "Good Health and Well-Being", "#4C9F38", "Ensure healthy lives and promote well-being for all at all ages"),
            # ... continue for all 17 SDGs
        ]
        
        for goal_num, name, color, desc in sdg_data:
            conn.execute(text("""
                INSERT INTO sdgs (goal_number, name, color_hex, description) 
                VALUES (:goal_num, :name, :color, :desc)
                ON CONFLICT (goal_number) DO NOTHING
            """), {"goal_num": goal_num, "name": name, "color": color, "desc": desc})
    
    logger.info("Database initialized with SDG data")

if __name__ == "__main__":
    initialize_database()
