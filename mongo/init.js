db.getSiblingDB("admin").createUser({
  user: "admin",
  pwd: "StrongPassword123!",
  roles: [ "root" ]
});