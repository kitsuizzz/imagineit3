from sqlalchemy import Column, Integer
from . import Base


class Images(Base):
    __tablename__ = "inferences"

    id = Column(Integer, primary_key=True, index=True)
