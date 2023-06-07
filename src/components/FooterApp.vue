<template>
  <footer className="mt-4 mb-4 flex justify-center">
          <div className="max-w-sm">
            <p className="text-center text-xs">
              <i>
                This app has ML embedded into it. Use at your own risk.
              </i>
              <br />
              <i>
                Refer to the <router-link to="/about">About page</router-link> for more information.
              </i>
            </p>
            <div className="mt-2 flex justify-center text-center">
              <div className="mr-1">App created by</div>
                <a href="https://www.linkedin.com/in/jacob-braswell/">
                  Jacob Braswell
                </a>          
            </div>
          </div>
        </footer>
  </template>

<script>
export default {
    name: "Footer",
};
</script>

<style scoped>
footer {
    background-color: #ffffffff;
    padding: 1rem;
    font-size: 0.9rem;
    text-align: center;
}
</style>