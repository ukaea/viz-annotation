import { Flex, Header, View } from "@adobe/react-spectrum";

/** Shown when the server answers 403: the resource exists, but this account has no
 * access to it. Kept separate from ErrorView so a missing project reads as
 * "not found" and an unauthorised one reads as "not yours".
 */
export default function ForbiddenView({ message }: { message?: string }) {
  return (
    <View width="100%">
      <Flex
        direction="column"
        gap="size-200"
        alignItems="center"
        marginTop="size-500"
      >
        <Header>
          <span style={{ fontSize: "15pt" }}>403 - Forbidden</span>
        </Header>
        <div style={{ fontSize: "48px", marginBottom: "16px" }}>🔒</div>
        <p style={{ color: "#666", maxWidth: "500px", textAlign: "center" }}>
          {message ||
            "You do not have access to this project. Ask a project admin to add you as a member."}
        </p>
      </Flex>
    </View>
  );
}
